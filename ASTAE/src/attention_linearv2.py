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
import copy
import collections


def pretrained_AST(input_tdim = 1024, audioset_only = False,audioset_pretrain=True,imagenet_pretrain=True,tiny=False):
    pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
    # get the frequency and time stride of the pretrained model from its name
    fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
    # initialize an AST model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("device: "+ str(device))
    #sd = torch.load(pretrained_mdl_path, map_location=device)
    #audio_model_ast = ast_model.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride,model_size='tiny224')
    #if audioset_only:
    #    pretrained_mdl_path = '../../audio-transformer/scv2/checkpoint-75.pth'

    if tiny:
          audio_model_ast = ast_model.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride,
                                             audioset_pretrain=False,imagenet_pretrain=False,model_size="tiny224")
          if audioset_pretrain and imagenet_pretrain:
             pretrained_mdl_path = '../../pretrained_models/tiny_ast_checkpoint-115.pth'
             sd = torch.load(pretrained_mdl_path, map_location=device)
             audio_model_ast.load_state_dict(sd, strict=False)

          else:
              raise ("not implemented")

    else:
        audio_model_ast = ast_model.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride,
                                             audioset_pretrain=audioset_pretrain,imagenet_pretrain=imagenet_pretrain)
    #audio_model_ast = torch.nn.DataParallel(audio_model_ast)
    #audio_model.load_state_dict(sd, strict=False)
    return audio_model_ast

def average_layers(output_model,models):
    state_dicts = []
    new_state_dict = collections.OrderedDict()
    for model in models:
        state_dicts.append(copy.deepcopy(model.state_dict()))
    for key in output_model.state_dict():
        param = torch.mean(torch.stack([sd[key] for sd in state_dicts]), dim=0)
        new_state_dict[key] = param
    output_model.load_state_dict(new_state_dict)


"""
Many parts are taken from AST: https://github.com/YuanGongND/ast
    parts copied from AST are annotated with #AST
and from ViT: timm.ast_model.vision_transformer.
    parts copied from ViT are annotated with #ViT
These modules only allow coarse selection of size (e.g. minimum of heads = 4).
The model is also different because we don't do classification.
This is why we don't use these modules directly, but copy the relevant parts instead.
"""

class Mlp(nn.Module):
    def __init__(self,in_features, out_features=None, act_layer=nn.GELU, drop=0.,requires_grad=True):
        super().__init__()
        out_features = out_features or in_features
        self.lin = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        if requires_grad:
            self.lin.requires_grad_(True)


    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
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


class Encoder(nn.Module):

    def __init__(self,depth_encoder,vit,trainable_encoder,avg=False,depth_trainable=0):
        super().__init__()
        if avg:
            averaged_encoder = copy.deepcopy(vit.blocks[0])
            average_layers(averaged_encoder,vit.blocks)
            self.encoderBlocks = torch.nn.ModuleList([averaged_encoder])
        else:
            self.encoderBlocks = vit.blocks[:depth_encoder]

        if trainable_encoder:
            self.encoderBlocks[-depth_trainable:].requires_grad_(True)
        self.vit = vit




    def forward(self, x):
        for blk in self.encoderBlocks: #AST
            x = blk(x) #AST
        x = self.vit.norm(x) #AST
        return x


class Decoder(nn.Module):

    def __init__(self,depth_decoder,feature_size,act_layer=nn.GELU,drop=0.,requires_grad=True):
        super().__init__()

        self.layers = [Mlp(feature_size, feature_size, act_layer, drop,requires_grad) for i in range(depth_decoder)]
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class attention_linear_model(nn.Module):
    """
    The AST AutoEncoder model.
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs,
        fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension,
        for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param depth_encoder: the number of blocks in the encoder
    :param depth_decoder: the number of layers in the decoder    """

    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, depth_encoder=1,depth_trainable=1,
                 depth_decoder=1,verbose=True, trainable_encoder = False,avg=False,audioset_only=False,
                 audioset_pretrain=True,imagenet_pretrain=True,tiny=False,dropout_decoder = 0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"

        print(device)

        super(attention_linear_model, self).__init__()
        assert timm.__version__ == '0.4.5', 'From AST model, not sure if really necessary'  # AST




        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('NETWORK SIZE:\n '
                  'encoder: depth {:d}'
                  'decoder: depth {:d}'
                  .format(depth_encoder,depth_decoder))

        # automatically get the intermediate shape # AST
        if tiny:
            self.embed_dim=192
        else:
            self.embed_dim=768
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        print("device: "+ str(device))
        self.AST_model = pretrained_AST(audioset_only=audioset_only,audioset_pretrain=audioset_pretrain,imagenet_pretrain=imagenet_pretrain,
                                        tiny=tiny)
        self.AST_model.requires_grad_(False)
        print("device: "+ str(device))
        #self.AST_model.v.to(device)
        self.patch_embed = self.AST_model.v.patch_embed
        self.patch_embed.requires_grad_(False)  # first prototype lin projection layer is untrained
        temp = self.AST_model.v.pos_embed
        self.pos_embed = nn.Parameter(self.AST_model.v.pos_embed[:,:-2,:]) # -2 as the model doesn't use distillation or class token
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))  # ViT
        self.pos_embed.requires_grad_(False)

        #norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #ViT
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth_encoder)]  # stochastic depth decay rule # vIt
        #self.encoderBlocks = self.AST_model.v.blocks

        self.encoder = Encoder(depth_encoder,self.AST_model.v,trainable_encoder,avg,depth_trainable)

        self.decoder = Decoder(depth_decoder,self.embed_dim,act_layer=nn.GELU,drop=dropout_decoder)
        self.decoder.requires_grad_(True)

        """
        self.encoderBlocks = nn.ModuleList([
            Encoder_block(
                dim=self.embed_dim, num_heads=num_heads_encoder, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth_encoder)])
        """



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

        B = x.shape[0]
        # untrained patch embed layer
        x = self.AST_model.v.patch_embed(x)
        cls_tokens = self.AST_model.v.cls_token.expand(B, -1, -1)
        dist_token = self.AST_model.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)



        x = x + self.AST_model.v.pos_embed #AST # dim (batch_size,nb_patches ,embedding dimension ), e.g., (32, 1212, 768)
        lin_proj_output = x.clone()
        lin_proj_output=(lin_proj_output[:, 0] + lin_proj_output[:, 1]) / 2
        #x = self.pos_drop(x) #AST  # we don't do dropout because pos embed is not trainable
        x = self.encoder(x)

        x = (x[:, 0] + x[:, 1]) / 2

        x = self.decoder(x)




        return x,lin_proj_output

