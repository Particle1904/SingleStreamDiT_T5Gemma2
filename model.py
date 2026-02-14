import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from functools import lru_cache
from config import Config

class Rope3D(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        
        self.dim_0 = head_dim // 4
        self.dim_1 = (head_dim - self.dim_0) // 2
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        
        assert self.dim_0 + self.dim_1 + self.dim_2 == head_dim

    def forward(self, seq_len_text, h, w, device):
        text_ids = torch.arange(seq_len_text, device=device, dtype=torch.float32).unsqueeze(1)
        zeros_text = torch.zeros_like(text_ids)
        grid_text = torch.cat([text_ids, zeros_text, zeros_text], dim=1) 

        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        
        flat_y = mesh_y.flatten().unsqueeze(1)
        flat_x = mesh_x.flatten().unsqueeze(1)
        time_img = torch.full_like(flat_y, seq_len_text) 
        grid_img = torch.cat([time_img, flat_y, flat_x], dim=1)
        grid = torch.cat([grid_text, grid_img], dim=0)
        
        def get_freqs(dim, positions):
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
            freqs = torch.outer(positions, inv_freq)
            return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

        freqs_0 = get_freqs(self.dim_0, grid[:, 0])
        freqs_1 = get_freqs(self.dim_1, grid[:, 1])
        freqs_2 = get_freqs(self.dim_2, grid[:, 2])
        
        return freqs_0, freqs_1, freqs_2

def apply_rope_3d(x, freqs_0, freqs_1, freqs_2):
    d0 = freqs_0.shape[-1]
    d1 = freqs_1.shape[-1]
    
    x0 = x[..., :d0]
    x1 = x[..., d0 : d0+d1]
    x2 = x[..., d0+d1:]
    
    def rotate_half(x, f):
        f = f.unsqueeze(0).unsqueeze(2)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = f[..., 0::2]
        s = f[..., 1::2]
        
        res = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return res

    x0 = rotate_half(x0, freqs_0)
    x1 = rotate_half(x1, freqs_1)
    x2 = rotate_half(x2, freqs_2)
    
    return torch.cat([x0, x1, x2], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class FourierFilter(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.low_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2, bias=False)
        )
        self.high_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2, bias=False)
        )
        self.raw_cutoff = nn.Parameter(torch.tensor(-1.0))
        self.raw_scale = nn.Parameter(torch.tensor(3.0))
        self.gate = nn.Parameter(torch.tensor([0.01]))
        
        nn.init.xavier_uniform_(self.low_mlp[-1].weight, gain=0.1)
        nn.init.xavier_uniform_(self.high_mlp[-1].weight, gain=0.1)
        nn.init.normal_(self.low_mlp[0].weight, std=0.02)
        nn.init.normal_(self.high_mlp[0].weight, std=0.02)

    @staticmethod
    @lru_cache(maxsize=32)
    def get_frequency_coords(h, w, device):
        fy = torch.fft.fftfreq(h, device=device)
        fx = torch.fft.rfftfreq(w, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        coords = torch.stack([gy, gx], dim=-1)
        return coords

    def forward(self, x, h, w):
        B, L, C = x.shape
        dtype = x.dtype
        device = x.device
        
        # 1. Reshape
        x_img = x.view(B, h, w, C).permute(0, 3, 1, 2).float()
        
        # 2. Reflection Padding
        pad = 2 
        x_padded = F.pad(x_img, (pad, pad, pad, pad), mode='reflect')
        h_pad, w_pad = h + (2 * pad), w + (2 * pad)
        
        # 3. Forward to Frequency Domain
        x_fft = torch.fft.rfft2(x_padded, dim=(2, 3), norm='ortho')
        h_freq, w_freq = x_fft.shape[2], x_fft.shape[3]
        
        # 4. Get Coordinates
        coords = self.get_frequency_coords(h_pad, w_pad, device) 
        coords_flat = coords.reshape(-1, 2).to(dtype)
        
        # 5. Anisotropic MLP
        raw_low = self.low_mlp(coords_flat)   
        raw_high = self.high_mlp(coords_flat) 
        
        # 6. Calculate Radius for Curriculum Mask
        r_flat = torch.norm(coords_flat, p=2, dim=-1, keepdim=True)
        r_flat = r_flat / 0.70710678
        
        # 7. Polar Math (Amplitude + Phase)
        amp_log_low, phase_low = raw_low.chunk(2, dim=-1)
        amp_log_high, phase_high = raw_high.chunk(2, dim=-1)
        
        cutoff = torch.sigmoid(self.raw_cutoff)
        scale = F.softplus(self.raw_scale)
        
        mask = torch.sigmoid((cutoff - r_flat) * scale)
        amp_log = mask * amp_log_low + (1.0 - mask) * amp_log_high
        phase_shift = mask * phase_low + (1.0 - mask) * phase_high
        
        # 8. Convert to Complex Filter Gain
        amp_gain = torch.exp(torch.tanh(amp_log))
        phase_shift = phase_shift * torch.pi 
        
        filter_real = amp_gain * torch.cos(phase_shift)
        filter_imag = amp_gain * torch.sin(phase_shift)
        complex_filter = torch.complex(filter_real, filter_imag)
        
        final_gain = complex_filter.view(h_freq, w_freq, C).permute(2, 0, 1).unsqueeze(0)
        
        # 9. Apply Filter in Frequency Domain
        x_fft_filtered = x_fft * final_gain
        
        # 10. Inverse FFT
        x_out = torch.fft.irfft2(x_fft_filtered, s=(h_pad, w_pad), dim=(2, 3), norm='ortho')
        
        # 11. Crop and Cleanup
        x_out = x_out[:, :, pad:pad+h, pad:pad+w]
        x_out = x_out.permute(0, 2, 3, 1).reshape(B, L, C)
        
        # 12. Gating
        return x_out.to(dtype) * torch.tanh(self.gate)

class VisualFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_fourier=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.attention_norm1 = RMSNorm(hidden_size)
        self.ffn_norm1 = RMSNorm(hidden_size)
        self.attention_norm_q = RMSNorm(self.head_dim)
        self.attention_norm_k = RMSNorm(self.head_dim)
        
        self.attention_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_out = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4)
        
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.fourier_filter = FourierFilter(hidden_size)
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, f0, f1, f2, img_h=None, img_w=None):
        B, N, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = self.attention_norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        q = self.attention_q(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        if f0 is not None:
            q = apply_rope_3d(q, f0, f1, f2)
            k = apply_rope_3d(k, f0, f1, f2)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        x = x + gate_msa.unsqueeze(1) * self.dropout(self.attention_out(attn))
        
        x_norm_ffn = self.ffn_norm1(x)
        x_modulated_ffn = x_norm_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.feed_forward(x_modulated_ffn)
        
        x = x + gate_mlp.unsqueeze(1) * self.dropout(ffn_out)
        
        if self.use_fourier and img_h is not None and img_w is not None:
            img_len = img_h * img_w
            img_start_idx = N - img_len
            
            img_tokens = x_modulated_ffn[:, img_start_idx:, :]
            fourier_res = self.fourier_filter(img_tokens, img_h, img_w)
            
            context_part = x[:, :img_start_idx, :]
            img_part = x[:, img_start_idx:, :] + fourier_res
            x = torch.cat([context_part, img_part], dim=1)
            
        return x

class ContextRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.attention_norm1 = RMSNorm(hidden_size)
        self.ffn_norm1 = RMSNorm(hidden_size)
        self.attention_norm_q = RMSNorm(self.head_dim)
        self.attention_norm_k = RMSNorm(self.head_dim)
        
        self.attention_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4)

    def forward(self, x, f0, f1, f2):
        B, N, C = x.shape
        x_norm = self.attention_norm1(x)
        
        q = self.attention_q(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        if f0 is not None:
            q = apply_rope_3d(q, f0, f1, f2)
            k = apply_rope_3d(k, f0, f1, f2)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        x = x + self.attention_out(attn)
        
        x_norm_ffn = self.ffn_norm1(x)
        x = x + self.feed_forward(x_norm_ffn)
        return x

class SingleStreamDiT(nn.Module):
    def __init__(self,
                 in_channels=Config.in_channels,
                 patch_size=Config.patch_size,
                 hidden_size=Config.hidden_size,
                 depth=Config.depth,
                 num_heads=Config.num_heads,
                 text_embed_dim=Config.text_embed_dim,
                 gradient_checkpointing=Config.gradient_checkpointing,
                 refiner_depth=Config.refiner_depth,
                 fourier_stack_depth=Config.fourier_stack_depth,
                 max_token_length=Config.max_token_length,
                 use_fourier_in_refiner=Config.use_fourier_filters_in_refiner,
                 dropout=Config.model_dropout,
                 rope_base=Config.rope_base):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.fourier_stack_depth = fourier_stack_depth
        self.use_fourier_in_refiner = use_fourier_in_refiner
        self.head_dim = hidden_size // num_heads
        
        patch_dim = in_channels * (patch_size ** 2)
        self.x_embedder = nn.Linear(patch_dim, hidden_size)
        self.cap_embedder = nn.Sequential(
            RMSNorm(text_embed_dim),
            nn.Linear(text_embed_dim, hidden_size, bias=True)
        )
        
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_token_length, hidden_size))
        self.x_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        self.t_embedder = nn.Sequential(
            nn.Linear(256, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.rope = Rope3D(self.head_dim, rope_base)
        
        self.noise_refiner = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_fourier=self.use_fourier_in_refiner) 
            for _ in range(refiner_depth)
        ])
        
        self.context_refiner = nn.ModuleList([
            ContextRefinerBlock(hidden_size, num_heads) 
            for _ in range(refiner_depth)
        ])
        
        self.blocks = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_fourier=False) 
            for _ in range(depth)
        ])
        
        self.final_fourier_blocks = nn.ModuleList()
        if self.fourier_stack_depth > 0:
            for _ in range(fourier_stack_depth):
                block = nn.ModuleDict({
                    'norm': RMSNorm(hidden_size),
                    'filter': FourierFilter(hidden_size)
                })
                self.final_fourier_blocks.append(block)
                
        self.final_norm = RMSNorm(hidden_size)
        self.final_layer = nn.Linear(hidden_size, patch_dim)
        
        self.initialize_weights()

    def forward(self, x, t, text_embeds):
        B, C, H, W = x.shape
        p = self.patch_size
        grid_h, grid_w = H // p, W // p
        
        # 1. Embed Inputs
        x = self.patchify(x)
        x = self.x_embedder(x)
        
        is_null = (text_embeds.abs().sum(dim=(1, 2)) == 0)
        context = self.cap_embedder(text_embeds)
        if is_null.any():
            null_mask = is_null.view(-1, 1, 1)
            context = torch.where(null_mask, self.cap_pad_token.expand_as(context), context)
            
        seq_len_text = context.shape[1]
        
        context = context + self.text_pos_embed[:, :seq_len_text, :]
        
        t_freq = self.timestep_embedding(t, 256)
        t_emb = self.t_embedder(t_freq.to(x.dtype))
        
        # 2. Generate 3D RoPE Frequencies
        f0, f1, f2 = self.rope(seq_len_text, grid_h, grid_w, x.device)
        
        # 3. Refiner Stages (Separate Streams)
        f0_txt, f0_img = f0[:seq_len_text], f0[seq_len_text:]
        f1_txt, f1_img = f1[:seq_len_text], f1[seq_len_text:]
        f2_txt, f2_img = f2[:seq_len_text], f2[seq_len_text:]
        
        for block in self.noise_refiner:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, t_emb, f0_img, f1_img, f2_img, grid_h, grid_w, use_reentrant=False)
            else:
                x = block(x, t_emb, f0_img, f1_img, f2_img, img_h=grid_h, img_w=grid_w)
                
        for block in self.context_refiner:
             if self.gradient_checkpointing:
                context = checkpoint(block, context, f0_txt, f1_txt, f2_txt, use_reentrant=False)
             else:
                context = block(context, f0_txt, f1_txt, f2_txt)
                
        # 4. Fusion
        x_concat = torch.cat([context, x], dim=1)
        
        for block in self.blocks:
            if self.gradient_checkpointing:
                x_concat = checkpoint(block, x_concat, t_emb, f0, f1, f2, grid_h, grid_w, use_reentrant=False)
            else:
                x_concat = block(x_concat, t_emb, f0, f1, f2, img_h=grid_h, img_w=grid_w)
                
        # 5. Output
        img_token_len = grid_h * grid_w
        x_out = x_concat[:, -img_token_len:, :]
        
        # 6. Final Fourier Stack (Global Refinement)
        if self.fourier_stack_depth > 0:
            current_x = x_out
            for block in self.final_fourier_blocks:
                norm_out = block['norm'](current_x)
                correction = block['filter'](norm_out, grid_h, grid_w)
                current_x = current_x + correction
            x_out = current_x
            
        x_out = self.final_norm(x_out)
        x_out = self.final_layer(x_out)
        x_out = self.unpatchify(x_out, grid_h, grid_w)
        
        return x_out

    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)
        return x

    def unpatchify(self, x, h, w):
        p = self.patch_size
        c = self.in_channels
        x = x.reshape(x.shape[0], h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], c, h * p, w * p)
        return x

    def initialize_weights(self):
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.constant_(self.cap_embedder[1].bias, 0)
        
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)
        
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)
        
        nn.init.normal_(self.x_pad_token, std=0.02)
        nn.init.normal_(self.cap_pad_token, std=0.02)
        nn.init.normal_(self.text_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
                
        for module in self.modules():
            if isinstance(module, SwiGLU):
                nn.init.xavier_uniform_(module.w1.weight)
                nn.init.xavier_uniform_(module.w2.weight)
                nn.init.xavier_uniform_(module.w3.weight)
                
        for module in [self.noise_refiner, self.context_refiner, self.blocks]:
            for p in module.parameters():
                if p.dim() > 1: nn.init.normal_(p, std=0.02)