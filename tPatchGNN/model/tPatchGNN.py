import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer

import lib.utils as utils
from lib.evaluation import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class nconv(nn.Module):
	def __init__(self):
		super(nconv,self).__init__()

	def forward(self, x, A):
		# x (B, F, N, M)
		# A (B, M, N, N)
		x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
		# print(x.shape)
		return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear,self).__init__()
		# self.mlp = nn.Linear(c_in, c_out)
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

	def forward(self, x):
		# x (B, F, N, M)

		# return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		return self.mlp(x)
		
class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn,self).__init__()
		self.nconv = nconv()
		c_in = (order*support_len+1)*c_in
		# c_in = (order*support_len)*c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		# x (B, F, N, M)
		# a (B, M, N, N)
		out = [x]
		for a in support:
			x1 = self.nconv(x,a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1,a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1) # concat x and x_conv
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
	

class tPatchGNN(nn.Module):
	def __init__(self, args, supports = None, dropout = 0):
	
		super(tPatchGNN, self).__init__()
		self.device = args.device
		self.hid_dim = args.hid_dim
		self.N = args.ndim
		self.M = args.npatch
		self.batch_size = None
		self.supports = supports
		self.n_layer = args.nlayer

		### Intra-time series modeling ## 
		## Time embedding
		self.te_scale = nn.Linear(1, 1)
		self.te_periodic = nn.Linear(1, args.te_dim-1)

		## TTCN
		input_dim = 1 + args.te_dim
		ttcn_dim = args.hid_dim - 1
		self.ttcn_dim = ttcn_dim
		self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
		self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
		
		d_model = args.hid_dim
		## Transformer
		self.ADD_PE = PositionalEncoding(d_model) 
		self.transformer_encoder = nn.ModuleList()
		for _ in range(self.n_layer):
			encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.nhead, batch_first=True)
			self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=args.tf_layer))			

		### Inter-time series modeling ###
		self.supports_len = 0
		if supports is not None:
			self.supports_len += len(supports)

		nodevec_dim = args.node_dim
		self.nodevec_dim = nodevec_dim
		if supports is None:
			self.supports = []

		self.nodevec1 = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)
		self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.N).cuda(), requires_grad=True)

		self.nodevec_linear1 = nn.ModuleList()
		self.nodevec_linear2 = nn.ModuleList()
		self.nodevec_gate1 = nn.ModuleList()
		self.nodevec_gate2 = nn.ModuleList()
		for _ in range(self.n_layer):
			self.nodevec_linear1.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_linear2.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_gate1.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
			self.nodevec_gate2.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
			
		self.supports_len +=1

		self.gconv = nn.ModuleList() # gragh conv
		for _ in range(self.n_layer):
			self.gconv.append(gcn(d_model, d_model, dropout, support_len=self.supports_len, order=args.hop))

		### Encoder output layer ###
		self.outlayer = args.outlayer
		enc_dim = args.hid_dim
		if(self.outlayer == "Linear"):
			self.temporal_agg = nn.Sequential(
				nn.Linear(args.hid_dim*self.M, enc_dim))
		
		elif(self.outlayer == "CNN"):
			self.temporal_agg = nn.Sequential(
				nn.Conv1d(d_model, enc_dim, kernel_size=self.M))

		### Decoder ###
		self.decoder = nn.Sequential(
			nn.Linear(enc_dim+args.te_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, 1)
			)
		
	def LearnableTE(self, tt):
		# tt: (N*M*B, L, 1)
		out1 = self.te_scale(tt)
		out2 = torch.sin(self.te_periodic(tt))
		return torch.cat([out1, out2], -1)
	
	def TTCN(self, X_int, mask_X):
		# X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)

		N, Lx, _ = mask_X.shape
		Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
		Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
		Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
		Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
		X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
		ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
		h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
		return h_t

	def IMTS_Model(self, x, mask_X):
		"""
		x (B*N*M, L, F)
		mask_X (B*N*M, L, 1)
		"""
		# mask for the patch
		mask_patch = (mask_X.sum(dim=1) > 0) # (B*N*M, 1)

		### TTCN for patch modeling ###
		x_patch = self.TTCN(x, mask_X) # (B*N*M, hid_dim-1)
		x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim)
		x_patch = x_patch.view(self.batch_size, self.N, self.M, -1) # (B, N, M, hid_dim)
		B, N, M, D = x_patch.shape

		x = x_patch
		for layer in range(self.n_layer):

			if(layer > 0): # residual
				x_last = x.clone()
				
			### Transformer for temporal modeling ###
			x = x.reshape(B*N, M, -1) # (B*N, M, F)
			x = self.ADD_PE(x)
			x = self.transformer_encoder[layer](x).view(x_patch.shape) # (B, N, M, F)

			### GNN for inter-time series modeling ###
			### time-adaptive graph structure learning ###
			nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
			nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)
			x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
			x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
			x_p1 = x_gate1 * self.nodevec_linear1[layer](x) # (B, M, N, 10)
			x_p2 = x_gate2 * self.nodevec_linear2[layer](x) # (B, M, N, 10)
			nodevec1 = nodevec1 + x_p1.permute(0,2,1,3) # (B, M, N, 10)
			nodevec2 = nodevec2 + x_p2.permute(0,2,3,1) # (B, M, 10, N)

			adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1) # (B, M, N, N) used
			new_supports = self.supports + [adp]

			# input x shape (B, F, N, M)
			x = self.gconv[layer](x.permute(0,3,1,2), new_supports) # (B, F, N, M)
			x = x.permute(0, 2, 3, 1) # (B, N, M, F)

			if(layer > 0): # residual addition
				x = x_last + x 

		### Output layer ###
		if(self.outlayer == "CNN"):
			x = x.reshape(self.batch_size*self.N, self.M, -1).permute(0, 2, 1) # (B*N, F, M)
			x = self.temporal_agg(x) # (B*N, F, M) -> (B*N, F, 1)
			x = x.view(self.batch_size, self.N, -1) # (B, N, F)

		elif(self.outlayer == "Linear"):
			x = x.reshape(self.batch_size, self.N, -1) # (B, N, M*F)
			x = self.temporal_agg(x) # (B, N, hid_dim)

		return x

	def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask = None):
		
		""" 
		time_steps_to_predict (B, L) [0, 1]
		X (B, M, L, N) 
		truth_time_steps (B, M, L, N) [0, 1]
		mask (B, M, L, N)

		To ====>

        X (B*N*M, L, 1)
		truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

		B, M, L_in, N = X.shape
		self.batch_size = B
		X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
		truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
		mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
		te_his = self.LearnableTE(truth_time_steps) # (B*N*M, L, F_te)

		X = torch.cat([X, te_his], dim=-1)  # (B*N*M, L, F)

		### *** a encoder to model irregular time series
		h = self.IMTS_Model(X, mask) # (B, N, hid_dim)

		""" Decoder """
		L_pred = time_steps_to_predict.shape[-1]
		h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1) # (B, N, Lp, F)
		time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1) # (B, N, Lp, 1)
		te_pred = self.LearnableTE(time_steps_to_predict) # (B, N, Lp, F_te)

		h = torch.cat([h, te_pred], dim=-1) # (B, N, Lp, F)

		# (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
		outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0) 

		return outputs # (1, B, Lp, N)

