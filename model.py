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
       
        self.dim_0 = (head_dim // 4) // 2 * 2
        rem = head_dim - self.dim_0
        self.dim_1 = (rem // 2) // 2 * 2
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
       
        assert self.dim_0 % 2 == 0 and self.dim_1 % 2 == 0 and self.dim_2 % 2 == 0
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
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
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
   
    def rotate(x_chunk, f):
        f = f.unsqueeze(0).unsqueeze(2)
        half = f.shape[-1] // 2
        cos = f[..., :half].to(x_chunk.dtype)
        sin = f[..., half:].to(x_chunk.dtype)
       
        x_even = x_chunk[..., ::2]
        x_odd  = x_chunk[..., 1::2]
       
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
       
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).reshape(x_chunk.shape)
   
    x0 = rotate(x0, freqs_0)
    x1 = rotate(x1, freqs_1)
    x2 = rotate(x2, freqs_2)
   
    return torch.cat([x0, x1, x2], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        var = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return self.weight * x_norm.to(x.dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256, use_conv=False):
        super().__init__()
        self.use_conv = use_conv 
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

        if self.use_conv:
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, 
                                    groups=hidden_dim, bias=True)

    def forward(self, x, seq_len_text=0, grid_h=0, grid_w=0):
        x1 = self.w1(x)
        x2 = self.w2(x)
        
        x_gated = F.silu(x1) * x2 

        if self.use_conv and grid_h > 0:
            text_part = x_gated[:, :seq_len_text, :]
            img_part = x_gated[:, seq_len_text:, :]
            
            B, L, C = img_part.shape
            img_spatial = img_part.transpose(1, 2).contiguous().view(B, C, grid_h, grid_w)
            img_spatial = self.dwconv(img_spatial)
            img_mixed = img_spatial.reshape(B, C, L).transpose(1, 2)
            
            x_gated = torch.cat([text_part, img_mixed], dim=1)
        
        return self.w3(x_gated)

class VisualFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_conv=False, use_xsa=True):
        super().__init__()
        self.use_xsa = use_xsa
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
        
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4.0, use_conv=use_conv)
            
        self.adaLN_msa = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        
        nn.init.constant_(self.adaLN_msa[-1].weight, 0); 
        nn.init.constant_(self.adaLN_msa[-1].bias, 0)
        
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0); 
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, f0, f1, f2, seq_len_text, grid_h, grid_w, attend_mask=None):
        B, N, C = x.shape
        
        if attend_mask is None:
            attend_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        shift_msa, scale_msa, gate_msa = self.adaLN_msa(c).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_mlp(c).chunk(3, dim=1)
        
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
        
        attn_mask = attend_mask.unsqueeze(1).unsqueeze(2)
        
        if self.use_xsa:
            if not hasattr(self, '_diag_mask') or self._diag_mask.shape[-1] != N:
                self._diag_mask = ~torch.eye(N, dtype=torch.bool, device=q.device).unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask & self._diag_mask
        
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        x = x + gate_msa.unsqueeze(1) * self.dropout(self.attention_out(attn))
        
        x_norm_ffn = self.ffn_norm1(x)
        x_modulated_ffn = x_norm_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.feed_forward(x_modulated_ffn, seq_len_text, grid_h, grid_w)
        
        x = x + gate_mlp.unsqueeze(1) * self.dropout(ffn_out)
            
        return x

class ContextRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, use_xsa=True):
        super().__init__()
        self.use_xsa = use_xsa
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
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4.0)

    def forward(self, x, f0, f1, f2, attend_mask=None):
        B, N, C = x.shape
        
        if attend_mask is None:
            attend_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
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
        
        attn_mask = attend_mask.unsqueeze(1).unsqueeze(2)
        
        if self.use_xsa:
            if not hasattr(self, '_diag_mask') or self._diag_mask.shape[-1] != N:
                self._diag_mask = ~torch.eye(N, dtype=torch.bool, device=q.device).unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask & self._diag_mask
        
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
                 max_token_length=Config.max_token_length,
                 dropout=Config.model_dropout,
                 rope_base=Config.rope_base,
                 use_xsa=Config.use_xsa):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.use_xsa = use_xsa
        
        patch_dim = in_channels * (patch_size ** 2)
        self.x_embedder = nn.Linear(patch_dim, hidden_size)
        self.cap_embedder = nn.Sequential(
            RMSNorm(text_embed_dim),
            nn.Linear(text_embed_dim, hidden_size, bias=True)
        )
        
        self.cap_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        self.t_embedder = nn.Sequential(
            nn.Linear(256, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.rope = Rope3D(self.head_dim, rope_base)
        
        self.noise_refiner = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_conv=True, use_xsa=self.use_xsa) 
            for _ in range(refiner_depth)
        ])
        
        self.context_refiner = nn.ModuleList([
            ContextRefinerBlock(hidden_size, num_heads, use_xsa=self.use_xsa) 
            for _ in range(refiner_depth)
        ])
        
        self.blocks = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_conv=True, use_xsa=self.use_xsa) 
            for _ in range(depth)
        ])            
                
        self.final_norm = RMSNorm(hidden_size)
        self.final_layer = nn.Linear(hidden_size, patch_dim)
                
        self.initialize_weights()

    def forward(self, x, t, text_embeds, text_mask=None):
        B, C, H, W = x.shape
        p = self.patch_size
        grid_h, grid_w = H // p, W // p
        
        # 1. Embed Inputs
        x = self.patchify(x)
        x = self.x_embedder(x)
        
        is_null = (text_embeds.abs().sum(dim=(1, 2)) == 0)
        context = self.cap_embedder(text_embeds)
        
        seq_len_text = context.shape[1]
        if text_mask is None:
            text_mask = torch.ones(B, seq_len_text, dtype=torch.bool, device=x.device)
        
        if is_null.any():
            null_mask = is_null.view(-1, 1, 1)
            context = torch.where(null_mask, self.cap_pad_token.expand_as(context), context)
        
        img_len = grid_h * grid_w
        full_mask = torch.cat([text_mask, torch.ones(B, img_len, dtype=torch.bool, device=x.device)], dim=1)
        
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
                x = checkpoint(block, x, t_emb, f0_img, f1_img, f2_img, 0, grid_h, grid_w,
                               attend_mask=None, use_reentrant=False)
            else:
                x = block(x, t_emb, f0_img, f1_img, f2_img, 0, grid_h, grid_w)
                
        for block in self.context_refiner:
            if self.gradient_checkpointing:
                context = checkpoint(block, context, f0_txt, f1_txt, f2_txt, 
                                     attend_mask=text_mask, use_reentrant=False)
            else:
                context = block(context, f0_txt, f1_txt, f2_txt, attend_mask=text_mask)
                
        # 4. Fusion
        x_concat = torch.cat([context, x], dim=1)
        
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x_concat = checkpoint(block, x_concat, t_emb, f0, f1, f2, seq_len_text, grid_h, 
                                      grid_w, attend_mask=full_mask, use_reentrant=False)
            else:
                x_concat = block(x_concat, t_emb, f0, f1, f2, seq_len_text, grid_h, grid_w, 
                                 attend_mask=full_mask)
            
        # 5. Output
        img_token_len = grid_h * grid_w
        x_out = x_concat[:, -img_token_len:, :]
            
        x_out = self.final_norm(x_out)
        x_out = self.final_layer(x_out)
        x_out = self.unpatchify(x_out, grid_h, grid_w)
        
        return x_out

    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        args = (t[:, None].float() * 1000.0) * freqs[None]
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
        
        nn.init.normal_(self.cap_pad_token, std=0.02)
        
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m not in [self.x_embedder, self.final_layer]:
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
            
            elif isinstance(m, SwiGLU):
                nn.init.xavier_uniform_(m.w1.weight)
                nn.init.xavier_uniform_(m.w2.weight)
                nn.init.xavier_uniform_(m.w3.weight)
                if m.use_conv:
                    with torch.no_grad():
                        m.dwconv.weight.fill_(0.0)
                        m.dwconv.weight[:, 0, 1, 1] = 1.0 
                        if m.dwconv.bias is not None:
                            nn.init.constant_(m.dwconv.bias, 0)