import math
#from re import X
from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.positionencoder import PositionEncoder
from models.mlp import MLP
from utils import make_coord
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv = DWConv(hidden_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x,x_size):
        x = self.fc1(x)
        x = self.act(x)

        x = self.dwconv(x,x_size)
        x = self.act(x)     #B C H W

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qk_scale = qk_scale or head_dim ** -0.5

        # mlp to genarate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(4, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=True))
        
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        
        # log
        # relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        #     torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, scale, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.qk_scale
        attn = (q @ k.transpose(-2, -1))

        # mlp
        rel_cell = scale.clone()
        rel_cell = rel_cell.unsqueeze(1).unsqueeze(2).repeat(1, 2*self.window_size[0]-1, 2*self.window_size[1]-1, 1)
        relative_coords  = torch.cat([self.relative_coords_table, rel_cell],dim=-1)
        relative_position_bias_table = self.cpb_mlp(relative_coords).view(-1, self.num_heads) 

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    


    
def Check_image_size(x, x_size, window_size):
    # x = x.permute(0, 3, 1, 2)
    H, W = x_size
    B, L, C = x.shape
    x = x.view(B, H, W, C).permute(0, 3, 1, 2)
    _, _, h, w = x.size()
    mod_pad_h = (h // window_size + 1) * window_size - h
    mod_pad_w = (w // window_size + 1) * window_size - w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    h_pad, w_pad = x.shape[2], x.shape[3]
    x = x.flatten(2).transpose(1, 2) 
    return x, h_pad, w_pad

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,out_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, size, scale):
        H, W = size

        x, h_pad, w_pad = Check_image_size(x, size, self.window_size)
        shortcut = x
        B, L, C = x.shape
        x = self.norm1(x)
        x = x.view(B, h_pad, w_pad, C)
        x_size = (h_pad, w_pad)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, scale, mask=self.attn_mask).to(x.device)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, scale, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        shifted_x = window_reverse(attn_windows, self.window_size, h_pad, w_pad)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # x = x.view(B, H * W, C)
        x = x.view(B, h_pad * w_pad, C)
        

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x),x_size))

        x = x.view(B, h_pad, w_pad, C)
        x = x[:, :H, :W, :]
        x = x.contiguous().view(B, H * W, C)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #self.reduction = nn.Linear(4 * dim,  dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        #H, W = self.input_resolution
        H, W = x_size[0], x_size[1]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, (x_size[0] // 2, x_size[1] // 2)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        #self.norm = norm_layer(dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        #H, W = self.input_resolution
        H, W = x_size[0], x_size[1]
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x, (x_size[0] * 2, x_size[1] * 2)

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, scale):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, scale)
            else:
                x = blk(x, x_size, scale)
        if self.downsample is not None:
            x, x_size = self.downsample(x, x_size)
        return x, x_size

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.upsample is not None:
            x , x_size= self.upsample(x, x_size)
        return x, x_size

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=64, embed_dim=192, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class Dconvbasic(nn.Module):
    def __init__(self, embed_dim = 192):
        super().__init__()
        self.embed_dim = embed_dim
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim),
            nn.PReLU()
        )


    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        x = self.dconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


#####crossswin
class CrossWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, window_size=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.x_qkv = nn.Linear(dim, dim , bias=qkv_bias)
        self.y_qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        B_Y, N_Y, C_Y = y.shape
        q = self.x_qkv(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_y = self.y_qkv(y).reshape(B_Y, N_Y, 2, self.num_heads, C_Y // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k_y, v_y = qkv_y[0], qkv_y[1]

        q = q * self.scale
        attn = (q @ k_y.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v_y).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class CrossSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution,num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        #self.y_input_resolution = y_input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        #H_Y, W_Y = y_size
        B, L, C = x.shape
        B_Y, L_Y, C_Y =y.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        y = self.norm1(y)
        y = y.view(B_Y, H, W, C_Y)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows,y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows,y_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x),x_size))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class CrossBlock(nn.Module):
    def __init__(self,dim, input_resolution, num_heads, depth, window_size=8, shift_size=0,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.swin = nn.ModuleList([
            CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        
    def forward(self, x, y, x_size):

        for blk in self.swin:
            if self.use_checkpoint:
                y = checkpoint.checkpoint(blk, x, y, x_size)
            else:
                y = blk(x, y, x_size)
        if self.downsample is not None:
            y ,x_size= self.downsample(y, x_size)
        return y

#Restormer CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, dim, y_dim, num_heads, bias):                 
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        # dim(192) is depth map channels num, y_dim(384) is color map channel num
        self.dim = dim                      
        self.y_dim = y_dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.y_qkv = nn.Conv2d(y_dim, y_dim*2, kernel_size=1, bias=bias)
        self.x_qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x_qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.y_qkv_dwconv = nn.Conv2d(y_dim*2, y_dim*2, kernel_size=3, stride=1, padding=1, groups=y_dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.dim, x_size[0], x_size[1])
        y = y.transpose(1, 2).view(B, self.y_dim, x_size[0], x_size[1])
        b,c,h,w = x.shape
        b_y, c_y, h_y, w_y = y.shape

        q = self.x_qkv_dwconv(self.x_qkv(x))
        kv = self.y_qkv_dwconv(self.y_qkv(y))
        k,v = kv.chunk(2, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c_y) h w -> b head c_y (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c_y) h w -> b head c_y (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = out.flatten(2).transpose(1, 2)
        return out

#Restormer CrossTransformerblock
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, y_dim,input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim                                       
        # dim(192) is depth map channels num, y_dim(384) is color map channel num
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm1_y = norm_layer(y_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = CrossAttention(dim, y_dim, num_heads,qkv_bias)

    def forward(self, x, y, x_size):
        H, W = x_size
        #H_Y, W_Y = y_size
        B, L, C = x.shape
        B_Y, L_Y, C_Y =y.shape
        # assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)

        y = self.norm1_y(y)
        x = self.attn(x, y, x_size) + shortcut
        x = x + self.mlp(self.norm2(x),x_size)
        return x


class FusionChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(FusionChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class LITLayer(nn.Module):
    def __init__(self,dim,out_dim,r=3,head=2,local_attn=True,verbose=False,is_cell=True,pb_spec='posenc',cell_decode=False):
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.local_attn = local_attn
        self.verbose = verbose
        self.head = head
        self.r = r
        self.is_cell = is_cell

        self.conv_vs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

        if self.local_attn:
            self.conv_qs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.conv_ks = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.is_pb = True if pb_spec else False

            if self.is_pb:
                self.pb_encoder = PositionEncoder(posenc_type='sinusoid', posenc_scale=10,
                                                  hidden_dims=64, enc_dims=64, head=self.head, gamma=1)
        else:
            self.r = 0

        self.r_area = (2 * self.r + 1)**2

        imnet_in_dim = self.dim + 2 if self.is_cell else self.dim 

        self.imnets = MLP(in_dim=imnet_in_dim, out_dim=self.out_dim, hidden_list=[256], act='gelu')

    def forward(self, x, coord, shape, cell):  
        
        x = self.query_rgb(x,sample_coord=coord, shape=shape, cell=cell)

        return x
        
        
    def query_rgb(self, feat, sample_coord, shape, cell=None):
        sample_coord = sample_coord.reshape(feat.shape[0],-1, 2)
        bs, q_sample, _ = sample_coord.shape                # b, q, 2
        device = sample_coord.device
        coord = make_coord(feat.shape[-2:], flatten=False).to(device=device).permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])                          #b, 2, h, w      
            
        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]  
        rw = 2 / feat.shape[-1]

        r = self.r    

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
                coord, sample_coord_.flip(-1), mode='nearest', align_corners=False
                        ).permute(0, 2, 3, 1)
            
        if self.local_attn:

            feat_q = self.conv_qs(feat)
            feat_k = self.conv_ks(feat)

            dh = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rh
            dw = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rw
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                        feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_q = sample_feat_q.reshape(
                    bs, q_sample, 1, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 2, 4)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                    feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_k = sample_feat_k.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 4, 2)
            
        feat_v = self.conv_vs(feat)
        sample_feat_v = F.grid_sample(
                feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)
            
        # b, q, 49, 2
        rel_coord = sample_coord_ - sample_coord_k     #
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        if self.local_attn:
            # b, q, h, 1, r_area -> b, q, r_area, h
            attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                    bs, q_sample, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

            if self.is_pb:
                _, pb = self.pb_encoder(rel_coord)
                attn = F.softmax(torch.add(attn, pb), dim=-2)
            else:
                attn = F.softmax(attn, dim=-2)

            attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1).permute(0,1,3,4,2)
            sample_feat_v = sample_feat_v.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0,1,3,2,4)
            # CA
            sample_feat_v = torch.matmul(attn,sample_feat_v).reshape(bs, q_sample, -1, self.dim // self.head)
            
        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets(feat_in)

        pred = pred + F.grid_sample(feat, sample_coord_.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        if self.local_attn and self.verbose:
            return pred, attn[0, :, :, :, 0].reshape(-1, 2 * r + 1, 2 * r + 1, self.head)
        return pred

class CrossLITLayer(nn.Module):
    def __init__(self,dim,out_dim,r=3,head=2,local_attn=True,verbose=False,is_cell=True,pb_spec='posenc',cell_decode=False):
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.local_attn = local_attn
        self.verbose = verbose
        self.head = head
        self.r = r
        self.is_cell = is_cell

        self.conv_vs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

        if self.local_attn:
            self.conv_qs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.conv_ks = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.is_pb = True if pb_spec else False

            if self.is_pb:
                self.pb_encoder = PositionEncoder(posenc_type='sinusoid', posenc_scale=10,
                                                  hidden_dims=64, enc_dims=64, head=self.head, gamma=1)
        else:
            self.r = 0

        self.r_area = (2 * self.r + 1)**2

        imnet_in_dim = self.dim + 2 if self.is_cell else self.dim 

        self.imnets = MLP(in_dim=imnet_in_dim, out_dim=self.out_dim, hidden_list=[256], act='gelu')

    def forward(self, x, depth, coord, shape, cell):  

        
        x = self.query_rgb(x, depth, sample_coord=coord, shape=shape, cell=cell)

        return x
        
        
    def query_rgb(self, feat, depth, sample_coord, shape, cell=None):
        # b, q, 2
        bs, q_sample, _ = sample_coord.shape
        device = sample_coord.device
        #b, 2, h, w
        coord = make_coord(feat.shape[-2:], flatten=False).to(device=device).permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])                              

        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]  
        rw = 2 / feat.shape[-1]

        r = self.r    

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
                coord, sample_coord_.flip(-1), mode='nearest', align_corners=False
                        ).permute(0, 2, 3, 1)
            
        if self.local_attn:
            feat_q = self.conv_qs(depth)
            feat_k = self.conv_ks(feat)

            dh = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rh
            dw = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rw
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                        feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_q = sample_feat_q.reshape(
                    bs, q_sample, 1, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 2, 4)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                    feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_k = sample_feat_k.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 4, 2)
            
        feat_v = self.conv_vs(feat)
        sample_feat_v = F.grid_sample(
                feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)
            
        # b, q, 49, 2
        rel_coord = sample_coord_ - sample_coord_k
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)

        rel_cell[..., 0] *= depth.shape[-2]
        rel_cell[..., 1] *= depth.shape[-1]

        if self.local_attn:
            # b, q, h, 1, r_area -> b, q, r_area, h
            attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                    bs, q_sample, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

            if self.is_pb:
                _, pb = self.pb_encoder(rel_coord)
                attn = F.softmax(torch.add(attn, pb), dim=-2)
            else:
                attn = F.softmax(attn, dim=-2)

            attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1).permute(0,1,3,4,2)
            sample_feat_v = sample_feat_v.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0,1,3,2,4)
            # CA
            sample_feat_v = torch.matmul(attn,sample_feat_v).reshape(bs, q_sample, -1, self.dim // self.head)
            
        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets(feat_in)

        pred = pred + F.grid_sample(depth, sample_coord_.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        if self.local_attn and self.verbose:
            return pred, attn[0, :, :, :, 0].reshape(-1, 2 * r + 1, 2 * r + 1, self.head)
        return pred  

class CrossLITFusion(nn.Module):
    def __init__(self,dim,out_dim,r=3,head=2,local_attn=True,verbose=False,is_cell=True,pb_spec='posenc',cell_decode=False):
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.local_attn = local_attn
        self.verbose = verbose
        self.head = head
        self.r = r
        self.is_cell = is_cell

        self.conv_vs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

        if self.local_attn:
            self.conv_qs = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.conv_ks = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)

            self.is_pb = True if pb_spec else False

            if self.is_pb:
                self.pb_encoder = PositionEncoder(posenc_type='cpb', head=self.head,
                                                  hidden_dims=64, enc_dims=64)

        else:
            self.r = 0

        self.r_area = (2 * self.r + 1)**2

        imnet_in_dim = self.dim + 2 if self.is_cell else self.dim

        self.imnets = MLP(in_dim=imnet_in_dim, out_dim=self.out_dim, hidden_list=[256], act='gelu')

    def forward(self, depth, x, shape_y, shape_x, cell):  
        
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, shape_y[0], shape_y[1])
        B, HW, C = depth.shape
        depth = depth.transpose(1, 2).view(B, C, shape_x[0], shape_x[1])
        x = self.query_rgb(x, depth, cell=cell)

        return x
        
        
    def query_rgb(self, feat, depth, cell=None):
        device = feat.device
        sample_coord = make_coord(depth.shape[-2:], flatten=False).to(device=device).unsqueeze(0).repeat(feat.shape[0], 1, 1, 1) 
        sample_coord = sample_coord.reshape(feat.shape[0],-1, 2) 
        bs, q_sample, _ = sample_coord.shape 
        
        coord = make_coord(feat.shape[-2:], flatten=False).to(device=device).permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])                          #b, 2, h, w    
            
        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]  
        rw = 2 / feat.shape[-1]
        # rh = 2 / shape[0]
        # rw = 2 / shape[1]

        r = self.r    

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
                coord, sample_coord_.flip(-1), mode='nearest', align_corners=False
                        ).permute(0, 2, 3, 1)                   # xq
            
        if self.local_attn:
            feat_q = self.conv_qs(depth)
            feat_k = self.conv_ks(feat)

            dh = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rh
            dw = torch.linspace(-r, r, 2 * r + 1).to(device=device) * rw
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                        feat_q, sample_coord_.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_q = sample_feat_q.reshape(
                    bs, q_sample, 1, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 2, 4)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                    feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
            sample_feat_k = sample_feat_k.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0, 1, 3, 4, 2)
            
        feat_v = self.conv_vs(feat)
        sample_feat_v = F.grid_sample(
                feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)
            
            # b, q, 49, 2
        rel_coord = sample_coord_ - sample_coord_k     #
        rel_coord[..., 0] *= depth.shape[-2]
        rel_coord[..., 1] *= depth.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        if self.local_attn:
            # b, q, h, 1, r_area -> b, q, r_area, h
            attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                    bs, q_sample, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

            if self.is_pb:
                _, pb = self.pb_encoder(rel_coord)
                attn = F.softmax(torch.add(attn, pb), dim=-2)
            else:
                attn = F.softmax(attn, dim=-2)

            attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1).permute(0,1,3,4,2)
            sample_feat_v = sample_feat_v.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                    ).permute(0,1,3,2,4)
            # CA
            sample_feat_v = torch.matmul(attn,sample_feat_v).reshape(bs, q_sample, -1, self.dim // self.head)
            
        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets(feat_in)

        # pred = pred + F.grid_sample(depth, sample_coord_.flip(-1), mode='bilinear',\
        #                         padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)
        pred = pred + depth.reshape(bs, self.out_dim, -1).permute(0, 2, 1)

        if self.local_attn and self.verbose:
            return pred, attn[0, :, :, :, 0].reshape(-1, 2 * r + 1, 2 * r + 1, self.head)
        return pred 

class Feature_Modulator(nn.Module):

    def __init__(self, in_dim, out_dim, act = True):
        super().__init__()

        self.act = nn.LeakyReLU()
        

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1),
            self.act
        )
        
    def gen_coord(self, in_shape, output_size):

        self.image_size =output_size
        self.coord      = make_coord(output_size,flatten=False) \
                            .expand(in_shape[0],output_size[0],output_size[1],2).flip(-1)      
        self.coord      = self.coord.to(self.device)

    def forward(self, guide_hr, feat, shape_y, shape_x):
        B, HW, C = guide_hr.shape
        guide_hr = guide_hr.transpose(1, 2).view(B, C, shape_y[0], shape_y[1])
        B, HW, C = feat.shape
        self.device = feat.device
        feat = feat.transpose(1, 2).view(B, C, shape_x[0], shape_x[1])
        self.gen_coord(feat.shape,(guide_hr.shape[2],guide_hr.shape[3]))
        q_feat = F.grid_sample(
            feat, self.coord, mode='bilinear', align_corners=False)
        guide_hr = F.grid_sample(
            guide_hr, self.coord, mode='bilinear', align_corners=False)

        q_feat = q_feat*guide_hr
        q_feat = self.conv1x1(q_feat)
        q_feat = q_feat.flatten(2).transpose(1, 2)

        return q_feat

class bi_model_lit(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, args ,img_size=32, img_size_y=128,patch_size=1, in_chans=1, out_chans=1,in_chans_y=3,
                 embed_dim=128, num_heads=[2,2], num_bif=2, depth=2,  
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, model_para = False,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.args = args
        self.out_chans = out_chans
        
        self.num_bif = num_bif
        self.num_depth = num_bif + 1
        self.num_guide = num_bif + 1
        depths = [depth for _ in range(self.num_depth)]
        guide_depths = [depth for _ in range(self.num_guide)]
        self.model_para = model_para
        
        # 存疑参数
        self.down_num = 4
        self.num_features = int(embed_dim)
        self.num_features_up = int(embed_dim * 2)
        self.final_upsample = final_upsample
        
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        
        self.mlp_ratio = mlp_ratio

        self.num_head = num_heads
        self.window_size = window_size
        
        self.depth_conv = nn.Conv2d(1, self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.rgb_conv = nn.Conv2d(3, self.embed_dim, kernel_size=3, stride=1, padding=1)

        # split image into non-overlapping patches
        self.patch_embed_depth = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_depth = self.patch_embed_depth.num_patches
        patches_resolution_depth = self.patch_embed_depth.patches_resolution
        self.patches_resolution_depth = patches_resolution_depth

        self.patch_embed_rgb = PatchEmbed(
            img_size=img_size_y, patch_size=patch_size, in_chans=self.embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_rgb = self.patch_embed_rgb.num_patches
        patches_resolution_rgb = self.patch_embed_rgb.patches_resolution
        self.patches_resolution_rgb = patches_resolution_rgb

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_depth = nn.Parameter(torch.zeros(1, num_patches_depth, embed_dim))
            trunc_normal_(self.absolute_pos_embed_depth, std=.02)
            self.absolute_pos_embed_rgb = nn.Parameter(torch.zeros(1, num_patches_rgb, embed_dim))
            trunc_normal_(self.absolute_pos_embed_rgb, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_d = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_g = [x.item() for x in torch.linspace(0, drop_path_rate, sum(guide_depths))]
   
        #CrossAttention
        self.bif = nn.ModuleList()
        for i_layer in range(self.num_bif):
            layer = nn.ModuleList([CrossLITFusion(dim=embed_dim,out_dim=embed_dim,r=3, head=num_heads[0],
                                                  local_attn=True,verbose=False,is_cell=True),
                                   Feature_Modulator(in_dim=embed_dim,out_dim=embed_dim)])
            self.bif.append(layer)
   
        # guidance image extract deep fearures
        self.guidedlayer = nn.ModuleList()
        for i_layer in range(self.num_guide):
            layer = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution_rgb[0],
                                                 patches_resolution_rgb[1]),
                               depth=guide_depths[i_layer],
                               num_heads=num_heads[0],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_g[sum(guide_depths[:i_layer]):sum(guide_depths[:i_layer+1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
            self.guidedlayer.append(layer)

        #depth map
        self.depthlayer = nn.ModuleList()
        for i_layer in range(self.num_depth):
            layer = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution_depth[0],
                                                 patches_resolution_depth[1]),
                               depth=depths[i_layer],
                               num_heads=num_heads[1],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_d[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
            self.depthlayer.append(layer) 
                           
        ###############color branch upsample and output##################
        self.upsample_rgb = CrossLITLayer(dim=embed_dim, out_dim=embed_dim, r=3, head=num_heads[0], local_attn=True, verbose=False, is_cell=True)
        # self.output_rgb = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1, padding=1)
        
        ##############depth branch upsample and output################
        self.upsample_depth = LITLayer(dim=embed_dim, out_dim=embed_dim, r=3, head=num_heads[1], local_attn=True, verbose=False, is_cell=True)

        #Fusion depth
        self.FusionOutput = nn.Linear(embed_dim*2, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (h // self.window_size + 1) * self.window_size - h
        mod_pad_w = (w // self.window_size + 1) * self.window_size - w
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
        
    def forward(self, data, eval_mode=False):
        # x is depth, y is color
        x, y = data['lr_depth'], data['hr_image']
        sample_coord = data['sample_coord']
        cell, lr_cell = data['cell'], data['lr_cell']
        scale = data['scale']
        
        # print(scale)

        x_lr = x
        H, W = y.shape[2], y.shape[3]
        B, hx, wx = x.shape[0], x.shape[2], x.shape[3]

        x_size  = (x.shape[2], x.shape[3])
        y_size  = (y.shape[2], y.shape[3])
        s = lr_cell.clone()
        s_hr = cell.clone()
        scale = cell.clone()
        s[...,0] *=  y.shape[2]/2
        s[...,1] *=  y.shape[3]/2
        s_hr[...,0] *=  x.shape[2]/2
        s_hr[...,1] *=  x.shape[3]/2 
        scale[...,0] *=  x.shape[2]
        scale[...,1] *=  x.shape[3]

        x = self.depth_conv(x)
        y = self.rgb_conv(y)

        x = self.patch_embed_depth(x)
        y = self.patch_embed_rgb(y)
        if self.ape:
            x = x + self.absolute_pos_embed_depth
            y = y + self.absolute_pos_embed_rgb
        x = self.pos_drop(x)
        y = self.pos_drop(y)
            
        x, x_size = self.depthlayer[0](x, x_size, scale)
        y, y_size = self.guidedlayer[0](y, y_size, scale)

        for i_layer in range(0, self.num_bif):
            x = self.bif[i_layer][0](x, y, y_size, x_size, lr_cell)
            x, x_size = self.depthlayer[i_layer+1](x, x_size, scale)
            y = self.bif[i_layer][1](y, x, y_size, x_size)
            y, y_size = self.guidedlayer[i_layer+1](y, y_size, scale)
        

        B, L, C = x.shape
        x = x.contiguous().view(B, hx, wx, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W

        y = y.contiguous().view(B, H, W, -1)
        y = y.permute(0, 3, 1, 2)  # B,C,H,W

        # train and eval
        if not eval_mode:
            z_xy = self.upsample_rgb(y, x, sample_coord, x_size, cell)
            z_x = self.upsample_depth(x, sample_coord, x_size, cell)
            sample_coord = sample_coord.unsqueeze(2)    # b,q,1,2

            z = torch.cat([z_xy,z_x],-1)
            z = self.FusionOutput(z)

            res = F.grid_sample(x_lr, sample_coord.flip(-1), mode='bicubic', padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)
        else:
            N = sample_coord.shape[1]
            n =  30720
            tmp_z_xy = []
            tmp_z_x = []
            for start in range(0, N, n):
                end = min(N, start + n)
                z_xy = self.upsample_rgb(y, x, sample_coord[:, start:end, :], x_size, cell)
                z_x = self.upsample_depth(x, sample_coord[:, start:end, :], x_size, cell)
                tmp_z_xy.append(z_xy)
                tmp_z_x.append(z_x)
            z_xy = torch.cat(tmp_z_xy, dim=1)
            z_x = torch.cat(tmp_z_x, dim=1)
            z = torch.cat([z_xy,z_x],-1)
            z = self.FusionOutput(z)
            sample_coord = sample_coord.unsqueeze(2) 
            res = F.grid_sample(x_lr, sample_coord.flip(-1), mode='bicubic', padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)
        return res, z + res
       
def make_model(args, parent=False):
    return bi_model_lit(args)
