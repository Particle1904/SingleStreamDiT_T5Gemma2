import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from functools import lru_cache
from config import Config

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class LocalSpatialBias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(
            dim, dim, 
            kernel_size=3, 
            padding=1, 
            groups=dim, 
            bias=True
        )
        nn.init.constant_(self.conv.weight, 1e-3)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, h, w):
        B, L, C = x.shape
        
        img_tokens = x[:, -h*w:, :] 
        feat_map = img_tokens.transpose(1, 2).reshape(B, C, h, w)
        
        local_feat = self.conv(feat_map)
        local_feat = local_feat.reshape(B, C, -1).transpose(1, 2)
        
        return local_feat

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
    def __init__(self, dim, expansion_factor=1.5):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.freq_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2)
        )
        
        self.gate = nn.Parameter(torch.zeros(1)) 

    def forward(self, x, h, w):
        B, L, C = x.shape
        dtype = x.dtype
        
        x_img = x.view(B, h, w, C)        
        x_img_fp32 = x_img.float()        
        x_fft = torch.fft.rfft2(x_img_fp32, dim=(1, 2), norm='ortho')   
             
        freq_real = torch.view_as_real(x_fft)         
        freq_flat = freq_real.flatten(start_dim=-2)  
               
        freq_processed = self.freq_mlp(freq_flat.float())      
        freq_processed = freq_processed.view(freq_real.shape)        
        freq_processed = torch.view_as_complex(freq_processed.float())   
             
        x_out = torch.fft.irfft2(freq_processed, s=(h, w), dim=(1, 2), norm='ortho')        
        x_out = x_out.reshape(B, L, C).to(dtype)
        
        return x_out * torch.tanh(self.gate)

@lru_cache(maxsize=32)
def create_2d_rope_grid(h, w, dim, device):
    grid_h = torch.arange(h, device=device)
    grid_w = torch.arange(w, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)
    
    inv_freq = 1.0 / (Config.rope_base ** (torch.arange(0, dim, 4, device=device).float() / dim))
    t_h = grid[:, 0:1] * inv_freq
    t_w = grid[:, 1:2] * inv_freq
    
    freqs = torch.cat([t_h, t_w], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    x_fp32 = x.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    d = x_fp32.shape[-1]
    x1 = x_fp32[..., :d//2]
    x2 = x_fp32[..., d//2:]
    rotated = torch.cat([x1 * cos_fp32 - x2 * sin_fp32, x1 * sin_fp32 + x2 * cos_fp32], dim=-1)
    return rotated.to(dtype=x.dtype)

class VisualFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
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

        self.fourier_filter = FourierFilter(hidden_size)
        self.fourier_norm = RMSNorm(hidden_size)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.local_bias = LocalSpatialBias(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, cos, sin, img_h=None, img_w=None):
        B, N, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = self.attention_norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        q = self.attention_q(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        
        img_len = img_h * img_w
        img_start_idx = N - img_len
        
        if cos is not None:
            q_img = q[:, :, img_start_idx:, :]
            k_img = k[:, :, img_start_idx:, :]
            
            if q_img.shape[2] == cos.shape[2]:
                q[:, :, img_start_idx:, :] = apply_rope(q_img, cos, sin)
                k[:, :, img_start_idx:, :] = apply_rope(k_img, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        gate_msa = gate_msa.tanh()
        gate_mlp = gate_mlp.tanh()
        
        x = x + gate_msa.unsqueeze(1) * self.dropout(self.attention_out(attn))
        
        if img_h is not None and img_w is not None:
            context_part = x[:, :-img_len, :]
            img_part = x[:, -img_len:, :]
            
            img_part = img_part + self.local_bias(img_part, img_h, img_w)
            x = torch.cat([context_part, img_part], dim=1)
        
        x_norm_ffn = self.ffn_norm1(x)
        x_modulated_ffn = x_norm_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        ffn_out = self.feed_forward(x_modulated_ffn)
        
        if img_h is not None and img_w is not None:
            img_tokens = x_modulated_ffn[:, img_start_idx:, :]
            
            fourier_input = self.fourier_norm(img_tokens)
            fourier_out = self.fourier_filter(fourier_input, img_h, img_w)
            
            ffn_out[:, img_start_idx:, :] = ffn_out[:, img_start_idx:, :] + fourier_out

        x = x + gate_mlp.unsqueeze(1) * self.dropout(ffn_out)
        
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

    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.attention_norm1(x)
        q = self.attention_q(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        x = x + self.attention_out(attn)
        x_norm_ffn = self.ffn_norm1(x)
        x = x + self.feed_forward(x_norm_ffn)
        return x

class SingleStreamDiTV2(nn.Module):
    def __init__(self, 
                 in_channels=Config.in_channels, 
                 patch_size=Config.patch_size, 
                 hidden_size=Config.hidden_size, 
                 depth=Config.depth, 
                 num_heads=Config.num_heads, 
                 text_embed_dim=Config.text_embed_dim, 
                 gradient_checkpointing=Config.gradient_checkpointing, 
                 refiner_depth=Config.refiner_depth, 
                 max_token_length=Config.max_token_length, 
                 dropout=Config.model_dropout):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        patch_dim = in_channels * (patch_size ** 2)
        
        self.x_embedder = nn.Linear(patch_dim, hidden_size)
        self.cap_embedder = nn.Sequential(RMSNorm(text_embed_dim),                                          
                                          nn.Linear(text_embed_dim, hidden_size, bias=True))
        
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_token_length, hidden_size))
        
        self.x_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Frequency embedding for Timesteps
        self.t_embedder = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        
        # Architecture components
        self.noise_refiner = nn.ModuleList([VisualFusionBlock(hidden_size, num_heads, dropout=dropout) for _ in range(refiner_depth)])
        self.context_refiner = nn.ModuleList([ContextRefinerBlock(hidden_size, num_heads) for _ in range(refiner_depth)])
        self.blocks = nn.ModuleList([VisualFusionBlock(hidden_size, num_heads, dropout=dropout) for _ in range(depth)])
        self.final_norm = RMSNorm(hidden_size)
        self.final_layer = nn.Linear(hidden_size, patch_dim)
        
        self.initialize_weights()

    def forward(self, x, t, text_embeds):
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Calculate dynamic grid for this batch
        grid_h, grid_w = H // p, W // p
        
        # 1. Patchify
        x = self.patchify(x)
        x = self.x_embedder(x)
        
        # 2. Text Embeddings
        is_null = (text_embeds.abs().sum(dim=(1, 2)) == 0)
        context = self.cap_embedder(text_embeds)
        
        if is_null.any():
            null_mask = is_null.view(-1, 1, 1)
            context = torch.where(null_mask, self.cap_pad_token.expand_as(context), context)
        
        seq_len = context.shape[1]
        context = context + self.text_pos_embed[:, :seq_len, :]
        
        # 3. Timestep
        t_freq = self.timestep_embedding(t, 256)
        t_emb = self.t_embedder(t_freq.to(x.dtype))
        
        # 4. RoPE Grid (Dynamic based on current bucket grid_h/w)
        cos, sin = create_2d_rope_grid(grid_h, grid_w, self.blocks[0].head_dim, x.device)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

        # 5. Refinement Phase (Separate streams)
        for block in self.noise_refiner:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, t_emb, cos, sin, grid_h, grid_w, use_reentrant=False)
            else:
                x = block(x, t_emb, cos, sin, img_h=grid_h, img_w=grid_w)
        
        for block in self.context_refiner:
             if self.gradient_checkpointing:
                context = checkpoint(block, context, use_reentrant=False)
             else:
                context = block(context)

        # 6. Joint Stream
        x_concat = torch.cat([context, x], dim=1)        
        for block in self.blocks:
            if self.gradient_checkpointing:
                x_concat = checkpoint(block, x_concat, t_emb, cos, sin, grid_h, grid_w, use_reentrant=False)
            else:
                x_concat = block(x_concat, t_emb, cos, sin, img_h=grid_h, img_w=grid_w)
        
        # 7. Unpatchify
        img_token_len = grid_h * grid_w
        x_out = x_concat[:, -img_token_len:, :]        
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
        # (B, C, H, W) -> (B, (H/p * W/p), C*p*p)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)
        return x
    
    def unpatchify(self, x, h, w):
        p = self.patch_size
        c = self.in_channels
        # (B, L, D) -> (B, C, H, W)
        x = x.reshape(x.shape[0], h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], c, h * p, w * p)
        return x
    
    def initialize_weights(self):
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.constant_(self.cap_embedder[1].bias, 0)
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)
        if self.t_embedder[0].bias is not None:
            nn.init.constant_(self.t_embedder[0].bias, 0)
        if self.t_embedder[2].bias is not None:
            nn.init.constant_(self.t_embedder[2].bias, 0)
        
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

        for block in self.noise_refiner:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for module in self.modules():
            if isinstance(module, FourierFilter):
                nn.init.xavier_uniform_(module.freq_mlp[0].weight)
                nn.init.constant_(module.freq_mlp[2].weight, 0)
                nn.init.constant_(module.freq_mlp[2].bias, 0)
                nn.init.constant_(module.gate, 0.1)