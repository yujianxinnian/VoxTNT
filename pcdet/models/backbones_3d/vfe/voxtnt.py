import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
import torch_scatter

import math
# import spconv
import spconv.pytorch as spconv

from torch_cluster import knn_graph
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VoxTNT(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.num_latents = self.model_cfg.NUM_LATENTS
        self.input_dim = self.model_cfg.INPUT_DIM
        self.output_dim = self.model_cfg.OUTPUT_DIM
        self.num_latents_2 = self.model_cfg.NUM_LATENTS_2

        self.input_embed = MLP(num_point_features, 16, self.input_dim, 2)
       
        self.pe0 = PositionalEncodingFourier(64, self.input_dim)
        self.pe1 = PositionalEncodingFourier(64, self.input_dim * 2)
        self.pe2 = PositionalEncodingFourier(64, self.input_dim * 4)
        self.pe3 = PositionalEncodingFourier(64, self.input_dim * 8)
    
        self.mlp_vsa_layer_0 = MLP_VSA_Layer(self.input_dim * 1, self.num_latents,self.num_latents[0],self.num_latents_2)
        self.mlp_vsa_layer_1 = MLP_VSA_Layer(self.input_dim * 2, self.num_latents,self.num_latents[1],self.num_latents_2)
        self.mlp_vsa_layer_2 = MLP_VSA_Layer(self.input_dim * 4, self.num_latents,self.num_latents[2],self.num_latents_2)
        self.mlp_vsa_layer_3 = MLP_VSA_Layer(self.input_dim * 8, self.num_latents,self.num_latents[3],self.num_latents_2)


        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )
        
       
        self.register_buffer('point_cloud_range', torch.FloatTensor(point_cloud_range).view(1, -1))
        self.register_buffer('voxel_size', torch.FloatTensor(voxel_size).view(1, -1))
        self.grid_size = grid_size.tolist()
        
        a, b, c = voxel_size
        self.register_buffer('voxel_size_02x', torch.FloatTensor([a * 2, b * 2, c]).view(1, -1))
        self.register_buffer('voxel_size_04x', torch.FloatTensor([a * 4, b * 4, c]).view(1, -1))
        self.register_buffer('voxel_size_08x', torch.FloatTensor([a * 8, b * 8, c]).view(1, -1))

        a, b, c = grid_size
        self.grid_size_02x = [a // 2, b //  2, c]
        self.grid_size_04x = [a // 4, b //  4, c]
        self.grid_size_08x = [a // 8, b //  8, c]
        
    def get_output_feature_dim(self):
        return self.output_dim

    
    def forward(self, batch_dict, **kwargs):
  
        points = batch_dict['points']
        points_offsets = points[:, 1:4] - self.point_cloud_range[:, :3]
        
        
        coords01x = points[:, :4].clone()
        coords01x[:, 1:4] = points_offsets // self.voxel_size
        pe_raw = (points_offsets - coords01x[:, 1:4] * self.voxel_size  ) / self.voxel_size
        coords01x, inverse01x = torch.unique(coords01x, return_inverse=True, dim=0)

        coords02x = points[:, :4].clone()
        coords02x[:, 1:4] = points_offsets // self.voxel_size_02x
        coords02x, inverse02x = torch.unique(coords02x, return_inverse=True, dim=0)
      
        coords04x = points[:, :4].clone()
        coords04x[:, 1:4] = points_offsets // self.voxel_size_04x
        coords04x, inverse04x = torch.unique(coords04x, return_inverse=True, dim=0)

        coords08x = points[:, :4].clone()
        coords08x[:, 1:4] = points_offsets // self.voxel_size_08x
        coords08x, inverse08x = torch.unique(coords08x, return_inverse=True, dim=0)


        src = self.input_embed(points[:, 1:]) 

        src = src + self.pe0(pe_raw)
        src = self.mlp_vsa_layer_0(src, inverse01x, coords01x, self.grid_size)

        src = src + self.pe1(pe_raw)
        src = self.mlp_vsa_layer_1(src, inverse02x, coords02x, self.grid_size_02x)

        src = src + self.pe2(pe_raw)
        src = self.mlp_vsa_layer_2(src, inverse04x, coords04x, self.grid_size_04x)

        src = src + self.pe3(pe_raw)
        src = self.mlp_vsa_layer_3(src, inverse08x, coords08x, self.grid_size_08x)

        src = self.post_mlp(src)

        batch_dict['point_features'] = F.relu(src)
        batch_dict['point_coords'] = points[:, :4]
        
        batch_dict['pillar_features'] = F.relu(torch_scatter.scatter_max(src, inverse01x, dim=0)[0])
        batch_dict['voxel_coords'] = coords01x[:, [0, 3, 2, 1]]
        
        return batch_dict


class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, num_latents=[8,8,8,8],n_latents=8,num_latents_2=[8,8,8,8]):
        super(MLP_VSA_Layer, self).__init__()
        self.dim = dim
        self.k = n_latents 
        self.num_latents_2 = num_latents_2 
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
        )

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)
      
        conv_dim = dim * self.k
        self.conv_dim = conv_dim


        self.tsfm_ffn = TSFM_FFN(conv_dim,dim,conv_dim,num_latents,self.num_latents_2)


    
        # decoder
        self.norm = nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True) 
        
    def forward(self, inp, inverse, coords, bev_shape):
        
        x = self.pre_mlp(inp) # (n,self.dim) eg.(83682,16)

        # encoder
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0) # (n,self.dim) eg.(83682,16)，计算注意力
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim*self.k) #(n,self.dim*self.k),eg.(n,128) x与注意力值乘积
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0) #(n',self.dim*self.k), eg.(13842, 128) n'体素个数，每个体素中点特征之和，也即体素局部特征呢个
        
        # 体素间信息交换 
        h = self.tsfm_ffn(F.relu(x_),coords) # # (n',self.dim*self.k)
        h = h[inverse, :] # (n,conv_dim),eg.[83682, 128]
       
        # decoder
        hs = self.norm(h.view(-1,  self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        
        # skip connection
        return torch.cat([inp, hs], dim=-1)
        
class TSFM_FFN(nn.Module):
    def __init__(self, num_voxel_features,input_dim,output_dim, num_latents=[8,8,8,8],num_latents_2=6):
        super(TSFM_FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents 
        self.num_latents_2 = num_latents_2

        self.input_embed = MLP(num_voxel_features, 16, input_dim, 2)
        self.pe0 = PositionalEncodingFourier(64, self.input_dim)
        self.pe1 = PositionalEncodingFourier(64, self.input_dim * 2)
        self.pe2 = PositionalEncodingFourier(64, self.input_dim * 4)
        self.pe3 = PositionalEncodingFourier(64, self.input_dim * 8)
 
        self.mlp_vsa_layer_0 = TSFM_FFN_VSA_Layer(self.input_dim * 1, self.num_latents_2[0])
        self.mlp_vsa_layer_1 = TSFM_FFN_VSA_Layer(self.input_dim * 2, self.num_latents_2[1])
        self.mlp_vsa_layer_2 = TSFM_FFN_VSA_Layer(self.input_dim * 4, self.num_latents_2[2])
        self.mlp_vsa_layer_3 = TSFM_FFN_VSA_Layer(self.input_dim * 8, self.num_latents_2[3])

        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )

    def forward(self, inp,coords):
        pe_raw = coords[:,1:].contiguous() #对应inp每个点/样本的坐标
        # 初始编码
        src = self.input_embed(inp) # (n,128)

        # 无GNN模式
        src = src + self.pe0(pe_raw) # (n,128)
        src = self.mlp_vsa_layer_0(src)

        src = src + self.pe1(pe_raw)
        src = self.mlp_vsa_layer_1(src)

        src = src + self.pe2(pe_raw)
        src = self.mlp_vsa_layer_2(src)

        src = src + self.pe3(pe_raw)
        src = self.mlp_vsa_layer_3(src)


        src = self.post_mlp(src)
        
        return src

class TSFM_FFN_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8):
        super(TSFM_FFN_VSA_Layer, self).__init__()
        self.dim = dim
        self.k = n_latents 
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
        )

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)
        
        
        # decoder
        self.norm = nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True) 
        
    def forward(self, inp):
        '''
            inp:(n,self.dim), eg.(83682,16),输入的特征张量，n为点的数量，16为特征维度
        '''
        #预处理
        x = self.pre_mlp(inp) #(n,self.dim),eg.(13842,128)
        # encoder
        attn = self.score(x) # self.score(x):(n,self.k) ,eg.(13842,8);
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim*self.k) #(n,self.dim*self.k),eg.(n,128*8) x与注意力值乘积 
        h = F.relu(dot)

        # decoder
        hs = self.norm(h.view(-1,  self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        
        # skip connection
        return torch.cat([inp, hs], dim=-1) 


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len
        
        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos   


