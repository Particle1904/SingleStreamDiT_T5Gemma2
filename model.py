import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from functools import lru_cache
from config import Config

class RoPE1D(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        freqs = torch.outer(positions, inv_freq)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

class RoPE2D(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, grid_h, grid_w, device):
        dim = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

        y = torch.arange(grid_h, device=device, dtype=torch.float32)
        x = torch.arange(grid_w, device=device, dtype=torch.float32)

        freqs_y = torch.outer(y, inv_freq)
        freqs_x = torch.outer(x, inv_freq)

        freqs_y = freqs_y.unsqueeze(1).expand(-1, grid_w, -1)
        freqs_x = freqs_x.unsqueeze(0).expand(grid_h, -1, -1)

        freqs = torch.cat([freqs_y, freqs_x], dim=-1)
        freqs = freqs.reshape(-1, dim)

        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def apply_rope(x, freqs):
    f = freqs.unsqueeze(0).unsqueeze(1)
    half = f.shape[-1] // 2
    cos = f[..., :half].to(x.dtype)
    sin = f[..., half:].to(x.dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]

    x_rot_1 = x1 * cos - x2 * sin
    x_rot_2 = x1 * sin + x2 * cos
    return torch.cat([x_rot_1, x_rot_2], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        var = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return (self.weight.to(x.dtype) * x_norm.to(x.dtype))
    
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
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_conv=False):
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
        
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4.0, use_conv=use_conv)
            
        self.adaLN_msa = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, freqs, seq_len_text, grid_h, grid_w, attend_mask=None):
        B, N, C = x.shape
        if attend_mask is None:
            attend_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            
        shift_msa, scale_msa, gate_msa = self.adaLN_msa(c).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_mlp(c).chunk(3, dim=1)
        
        x_norm = self.attention_norm1(x)
        if seq_len_text > 0:
            x_txt = x_norm[:, :seq_len_text]
            x_img = x_norm[:, seq_len_text:]
            x_img_mod = x_img * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            x_modulated = torch.cat([x_txt, x_img_mod], dim=1)
        else:
            x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        q = self.attention_q(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_modulated).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
            
        attn_mask = attend_mask.unsqueeze(1).unsqueeze(2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        x = x + gate_msa.unsqueeze(1) * self.dropout(self.attention_out(attn))
        
        x_norm_ffn = self.ffn_norm1(x)
        if seq_len_text > 0:
            x_txt_ffn = x_norm_ffn[:, :seq_len_text]
            x_img_ffn = x_norm_ffn[:, seq_len_text:]
            x_img_mod_ffn = x_img_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            x_modulated_ffn = torch.cat([x_txt_ffn, x_img_mod_ffn], dim=1)
        else:
            x_modulated_ffn = x_norm_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            
        ffn_out = self.feed_forward(x_modulated_ffn, seq_len_text, grid_h, grid_w)
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
        self.feed_forward = SwiGLU(hidden_size, hidden_size * 4.0)

    def forward(self, x, freqs, attend_mask=None):
        B, N, C = x.shape
        
        if attend_mask is None:
            attend_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        x_norm = self.attention_norm1(x)
        
        q = self.attention_q(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention_k(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention_v(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q = self.attention_norm_q(q)
        k = self.attention_norm_k(k)
        
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
                    
        attn_mask = attend_mask.unsqueeze(1).unsqueeze(2)
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
                 rope_base=Config.rope_base):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
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
        
        self.rope_txt = RoPE1D(self.head_dim, rope_base)
        self.rope_img = RoPE2D(self.head_dim, rope_base)
        
        self.noise_refiner = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_conv=True) 
            for _ in range(refiner_depth)
        ])
        
        self.context_refiner = nn.ModuleList([
            ContextRefinerBlock(hidden_size, num_heads) 
            for _ in range(refiner_depth)
        ])
        
        self.blocks = nn.ModuleList([
            VisualFusionBlock(hidden_size, num_heads, dropout=dropout, use_conv=True) 
            for _ in range(depth)
        ])            
                
        self.final_norm = RMSNorm(hidden_size)
        self.final_layer = nn.Linear(hidden_size, patch_dim)
                
        self.initialize_weights()

    def forward(self, x, t, text_embeds, text_mask=None):
        B, C, H, W = x.shape
        p = self.patch_size
        grid_h, grid_w = H // p, W // p

        x = self.patchify(x)
        x = self.x_embedder(x)
        
        is_null = (text_embeds.abs().sum(dim=(1, 2)) == 0)
        context = self.cap_embedder(text_embeds)
        
        if is_null.any():
            null_mask = is_null.view(-1, 1, 1)
            context = torch.where(null_mask, self.cap_pad_token.expand_as(context), context)
            
        seq_len_text = context.shape[1]
        
        if text_mask is None:
            text_mask = torch.ones(B, seq_len_text, dtype=torch.bool, device=x.device)
        else:
            text_mask = text_mask.bool()
            
        img_len = grid_h * grid_w
        full_mask = torch.cat([text_mask, torch.ones(B, img_len, dtype=torch.bool, device=x.device)], dim=1)
        
        t_freq = self.timestep_embedding(t, 256)
        t_emb = self.t_embedder(t_freq.to(x.dtype))

        freqs_txt = self.rope_txt(seq_len_text, x.device)
        freqs_img = self.rope_img(grid_h, grid_w, x.device)
        freqs = torch.cat([freqs_txt, freqs_img], dim=0)

        for block in self.noise_refiner:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, t_emb, freqs_img, 0, grid_h, grid_w,
                               None, use_reentrant=False)
            else:
                x = block(x, t_emb, freqs_img, 0, grid_h, grid_w)

        for block in self.context_refiner:
            if self.gradient_checkpointing:
                context = checkpoint(block, context, freqs_txt, text_mask, use_reentrant=False)
            else:
                context = block(context, freqs_txt, attend_mask=text_mask)

        x_concat = torch.cat([context, x], dim=1)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x_concat = checkpoint(block, x_concat, t_emb, freqs, seq_len_text, grid_h, 
                                      grid_w, full_mask, use_reentrant=False)
            else:
                x_concat = block(x_concat, t_emb, freqs, seq_len_text, grid_h, grid_w, 
                                 attend_mask=full_mask)

        x_out = x_concat[:, -img_len:, :]
        x_out = self.final_norm(x_out)
        x_out = self.final_layer(x_out)
        x_out = self.unpatchify(x_out, grid_h, grid_w)
        x_out = x_out + (self.cap_pad_token.sum() * 0.0).to(x_out.dtype)
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
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

        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.constant_(self.cap_embedder[1].bias, 0)
        
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)
        
        nn.init.normal_(self.cap_pad_token, std=0.02)
        
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)
                            
        for block in self.noise_refiner + self.blocks:
            nn.init.constant_(block.adaLN_msa[-1].weight, 0)
            nn.init.constant_(block.adaLN_msa[-1].bias, 0)
            nn.init.constant_(block.adaLN_mlp[-1].weight, 0)
            nn.init.constant_(block.adaLN_mlp[-1].bias, 0)
            
        for block in self.context_refiner:
            nn.init.constant_(block.attention_out.weight, 0)
            nn.init.constant_(block.feed_forward.w3.weight, 0)