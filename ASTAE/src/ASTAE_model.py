# AST AutoEncoder
import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__, "../../../"))) + '/src'
print(parentdir)
sys.path.append(parentdir)
import ast_model

import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_,StdConv2dSame, DropPath, trunc_normal_
from torch.cuda.amp import autocast
from functools import partial



def pretrained_AST(input_tdim = 1024):
    pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
    # get the frequency and time stride of the pretrained model from its name
    fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
    # initialize an AST model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("device: "+ str(device))
    sd = torch.load(pretrained_mdl_path, map_location=device)
    #audio_model_ast = ast_model.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride,model_size='tiny224')
    audio_model_ast = ast_model.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride)
    #audio_model = torch.nn.DataParallel(audio_model_ast)
    #audio_model.load_state_dict(sd, strict=False)
    return audio_model_ast




"""
Many parts are taken from AST: https://github.com/YuanGongND/ast
    parts copied from AST are annotated with #AST
and from ViT: timm.ast_model.vision_transformer.
    parts copied from ViT are annotated with #ViT
These modules only allow coarse selection of size (e.g. minimum of layers = 4).
The model is also different because we don't do classification.
This is why we don't use these modules directly, but copy the relevant parts instead.
Another source of inspiration is 'Attention is all you need' directly,
with code examples from https://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
"""

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.query_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_lin = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query,key,value,mask):
        B, N, C = query.shape # dim (batch_size, nb_patches, embedding_size), e.g., (32, 1212, 768)
        q=self.query_lin(query) # dim (batch_size, nb_patches, embedding_size), e.g., (32, 1212, 768)
        q = q.reshape(B,self.num_heads,-1,C//self.num_heads)
        k=self.key_lin(key).reshape(B,self.num_heads,-1,C//self.num_heads)
        v=self.value_lin(value).reshape(B,self.num_heads,-1,C//self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = attn.masked_fill(mask == 0, -1e9)
        # zero mask before softmax won't work, best solution is very large neg value:
        # see https://github.com/huggingface/transformers/issues/542
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Self_attention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

    def forward(self, x):
        super().forward(x,x,x,None)
        return x

class Encoder_block(nn.Module): #ViT

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Self_attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Decoder_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.self_attn = Self_attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,memory,mask):
        x= self.norm1(x)
        x = x + self.drop_path(self.attn())
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.drop_path(self.attn(x,memory,memory,mask))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ASTAEModel(nn.Module):
    """
    The AST AutoEncoder model.
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs,
        fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension,
        for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param depth_encoder: the number of blocks in the encoder
    :param depth_decoder: the number of blocks in the decoder
    :param num_heads_encoder: the number of heads in the encoder
    :param num_heads_decoder: the number of heads in the decoder

    """

    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, depth_encoder=1,
                 depth_decoder=4,num_heads_encoder=4,num_heads_decoder=4,drop_rate=0., mlp_ratio=4.,
                 qkv_bias=True,qk_scale=None,attn_drop_rate=0.,drop_path_rate=0.,norm_layer=None,verbose=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"


        super(ASTAEModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'From AST model, not sure if really necessary'  # AST




        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('NETWORK SIZE:\n '
                  'encoder: depth {:d} with {:d} heads '
                  'decoder: depth {:d} with {:d} heads'
                  .format(depth_encoder,num_heads_encoder,depth_decoder, num_heads_decoder))

        # automatically get the intermediate shape # AST
        self.embed_dim=768
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        print("device: "+ str(device))
        self.AST_model = pretrained_AST()
        self.AST_model.requires_grad_(False)
        print("device: "+ str(device))
        #self.AST_model.v.to(device)
        self.patch_embed = self.AST_model.v.patch_embed
        #self.patch_embed.requires_grad_(True)  # first prototype lin projection layer is untrained

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))  # ViT

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #ViT
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth_encoder)]  # stochastic depth decay rule # vIt
        self.encoderBlocks = nn.ModuleList([
            Encoder_block(
                dim=self.embed_dim, num_heads=num_heads_encoder, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth_encoder)])

        """self.decoderBlocks = nn.ModuleList([
            Decoder_block(
                dim=self.embed_dim, num_heads=num_heads_decoder, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth_decoder)])"""


    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):  # AST
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (32, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (32, 1024, 128)
        x = x.unsqueeze(1)  # dim (batch_size, 1, time_frame_num, frequency_bins), e.g., (32, 1, 1024, 128)
        x = x.transpose(2, 3)  # dim (batch_size, 1, frequency_bins,time_frame_num, ), e.g., (32, 1, 128, 1024)

        # untrained patch embed layer
        x = self.patch_embed(x)
        lin_proj_output = x.clone()

        x = x + self.pos_embed #AST # dim (batch_size,nb_patches ,embedding dimension ), e.g., (32, 1212, 768)
        #x = self.pos_drop(x) #AST  # we don't do dropout because pos embed is not trainable
        for blk in self.encoderBlocks: #AST
            x = blk(x) #AST
        x = self.AST_model.v.norm(x) #AST


        return x,lin_proj_output

