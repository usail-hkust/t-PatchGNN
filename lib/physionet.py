import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device

# Adapted from: https://github.com/rtqichen/time-series-datasets

class PhysioNet(object):

	urls = [
		'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
		'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
		'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
	]

	# outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

	params = [
		'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
		'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
		'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
		'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
	]

	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
	labels_dict = {k: i for i, k in enumerate(labels)}

	def __init__(self, root, download = False,
		quantization = None, n_samples = None, device = torch.device("cpu")):

		self.root = root
		# self.train = train
		self.reduce = "average"
		self.quantization = quantization

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')

		# if self.train:
		# 	data_file = self.training_file
		# else:
		# 	data_file = self.test_file
		
		if device == torch.device("cpu"):
			# self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
			# self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
			data_a = torch.load(os.path.join(self.processed_folder, self.set_a), map_location='cpu')
			data_b = torch.load(os.path.join(self.processed_folder, self.set_b), map_location='cpu')
			data_c = torch.load(os.path.join(self.processed_folder, self.set_c), map_location='cpu')
		else:
			data_a = torch.load(os.path.join(self.processed_folder, self.set_a))
			data_b = torch.load(os.path.join(self.processed_folder, self.set_b))
			data_c = torch.load(os.path.join(self.processed_folder, self.set_c))
			# self.data = torch.cat([data_a, data_b, data_c], dim=0)
			# self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))
			# print(len(self.data), len(self.labels.shape))

		self.data = data_a + data_b + data_c # a list with length 12000

		if n_samples is not None:
			print('Total records:', len(self.data))
			self.data = self.data[:n_samples]
			# self.labels = self.labels[:n_samples]


	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		# Download outcome data
		# for url in self.outcome_urls:
		# 	filename = url.rpartition('/')[2]
		# 	download_url(url, self.raw_folder, filename, None)

		# 	txtfile = os.path.join(self.raw_folder, filename)
		# 	with open(txtfile) as f:
		# 		lines = f.readlines()
		# 		outcomes = {}
		# 		for l in lines[1:]:
		# 			l = l.rstrip().split(',')
		# 			record_id, labels = l[0], np.array(l[1:]).astype(float)
		# 			outcomes[record_id] = torch.Tensor(labels).to(self.device)

		# 		torch.save(
		# 			labels,
		# 			os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
		# 		)

		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)
			tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
			tar.extractall(self.raw_folder)
			tar.close()

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder, filename.split('.')[0])
			patients = []
			total = 0
			cnt = 0
			for txtfile in os.listdir(dirname):
				record_id = txtfile.split('.')[0]
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = 0
					tt = [0.]
					vals = [torch.zeros(len(self.params))]
					mask = [torch.zeros(len(self.params))]
					nobs = [torch.zeros(len(self.params))]
					for l in lines[1:]:
						total += 1
						time, param, val = l.split(',')
						# Time in hours
						time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.

						# round up the time stamps (up to 6 min by default)
						# used for speed -- we actually don't need to quantize it in Latent ODE
						if(self.quantization != None and self.quantization != 0):
							time = round(time / self.quantization) * self.quantization

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.params)))
							mask.append(torch.zeros(len(self.params)))
							nobs.append(torch.zeros(len(self.params)))
							prev_time = time

						if param in self.params_dict:
							#vals[-1][self.params_dict[param]] = float(val)
							n_observations = nobs[-1][self.params_dict[param]]
							if self.reduce == 'average' and n_observations > 0:
								prev_val = vals[-1][self.params_dict[param]]
								new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
								vals[-1][self.params_dict[param]] = new_val
							else:
								vals[-1][self.params_dict[param]] = float(val)
							mask[-1][self.params_dict[param]] = 1
							nobs[-1][self.params_dict[param]] += 1
						else:
							assert (param == 'RecordID' or param ==''), 'Read unexpected param {}'.format(param)
							if(param != 'RecordID'):
								cnt += 1
								print(cnt, param, l)

				tt = torch.tensor(tt).to(self.device)
				vals = torch.stack(vals).to(self.device)
				mask = torch.stack(mask).to(self.device)

				# labels = None
				# if record_id in outcomes:
				# 	# Only training set has labels
				# 	labels = outcomes[record_id]
				# 	# Out of 5 label types provided for Physionet, take only the last one -- mortality
				# 	labels = labels[4]

				patients.append((record_id, tt, vals, mask))

			torch.save(
				patients,
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			)
				
		print('Done!')

	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]

			if not os.path.exists(
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	# @property
	# def training_file(self):
	# 	return 'set-a_{}.pt'.format(self.quantization)

	# @property
	# def test_file(self):
	# 	return 'set-b_{}.pt'.format(self.quantization)

	@property
	def set_a(self):
		return 'set-a_{}.pt'.format(self.quantization)

	@property
	def set_b(self):
		return 'set-b_{}.pt'.format(self.quantization)
	
	@property
	def set_c(self):
		return 'set-c_{}.pt'.format(self.quantization)

	# @property
	# def label_file(self):
	# 	return 'Outcomes-a.pt'

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Quantization: {}\n'.format(self.quantization)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str

	def visualize(self, timesteps, data, mask, plot_name):
		width = 15
		height = 15

		non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		mask = mask[:, non_zero_idx]
		data = data[:, non_zero_idx]
		
		params_non_zero = [self.params[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		n_col = 3
		n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
		fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		#for i in range(len(self.params)):
		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			tp_mask = mask[:,param_id].long()

			tp_cur_param = timesteps[tp_mask == 1.]
			data_cur_param = data[tp_mask == 1., param_id]

			ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o') 
			ax_list[i // n_col, i % n_col].set_title(param)

		fig.tight_layout()
		fig.savefig(plot_name)
		plt.close(fig)


# get minimum and maximum for each feature across the whole dataset
# def get_data_min_max(records, device):
# 	data_min, data_max, time_max = None, None, torch.tensor(0)

# 	inf = torch.Tensor([float("Inf")])[0].to(device)

# 	for b, (record_id, tt, vals, mask) in enumerate(records):
# 		n_features = vals.size(-1)

# 		batch_min = []
# 		batch_max = []
# 		for i in range(n_features):
# 			non_missing_vals = vals[:,i][mask[:,i] == 1]
# 			if len(non_missing_vals) == 0:
# 				batch_min.append(inf)
# 				batch_max.append(-inf)
# 			else:
# 				batch_min.append(torch.min(non_missing_vals))
# 				batch_max.append(torch.max(non_missing_vals))

# 		batch_min = torch.stack(batch_min)
# 		batch_max = torch.stack(batch_max)

# 		if (data_min is None) and (data_max is None):
# 			data_min = batch_min
# 			data_max = batch_max
# 		else:
# 			data_min = torch.min(data_min, batch_min)
# 			data_max = torch.max(data_max, batch_max)

# 		time_max = torch.max(time_max, tt.max())

# 	return data_min, data_max, time_max


def get_data_min_max(records, device):
	inf = torch.Tensor([float("Inf")])[0].to(device)

	data_min, data_max, time_max = None, None, -inf

	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

		time_max = torch.max(time_max, tt.max())

	# data_min = torch.where(torch.isinf(data_min), 1., data_min)
	# data_max = torch.where(torch.isinf(data_max), 1., data_max)

	print('data_max:', data_max)
	print('data_min:', data_min)
	print('time_max:', time_max)

	return data_min, data_max, time_max

def get_seq_length(args, records):
	
	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_observed_tp = torch.lt(tt, args.history).sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len

# def get_median_len(args, records):
# 	lens = []
# 	for b, (record_id, tt, vals, mask) in enumerate(records):
# 		n_observed_tp = torch.lt(tt, args.history).sum()
# 		lens.append(mask[:n_observed_tp].sum(dim=0)) # (ndim, )
# 		print(lens)
# 	lens = torch.cat(lens, dim=0)
# 	median_len = lens.median()
# 	print('median_len:', median_len)
# 	return median_len

# def patch_variable_time_collate_fn_bak(batch, args, device = torch.device("cpu"), data_type = "train", 
# 	data_min = None, data_max = None, time_max = None):
# 	"""
# 	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
# 		- record_id is a patient id
# 		- tt is a (T, ) tensor containing T time values of observations.
# 		- vals is a (T, D) tensor containing observed values for D variables.
# 		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
# 	Returns:
# 	Data form as input:
# 		batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
# 		batch_vals: (B, M, L_in, D) tensor containing the observed values.
# 		batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
# 	Data form to predict:
# 		flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
# 		flat_vals: (B, L_out, D) tensor containing the observed values.
# 		flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
# 	"""

# 	D = batch[0][2].shape[1]
# 	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
# 	# combined_tt = combined_tt.to(device)
# 	# print(combined_tt.shape)
# 	# print(inverse_indices.shape, np.sum([len(ex[1]) for ex in batch]), inverse_indices.max())
# 	# print(inverse_indices)

# 	# the number of observed time points 
# 	n_observed_tp = torch.lt(combined_tt, args.history).sum()
# 	observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )
# 	# print(n_observed_tp, len(combined_tt)-n_observed_tp)
# 	# print(combined_tt[:n_observed_tp])
# 	# print(combined_tt[n_observed_tp:])

# 	patch_indices = []
# 	st, ed = 0, args.patch_size
# 	for i in range(args.npatch):
# 		if(i == args.npatch-1):
# 			inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
# 		else:
# 			inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
# 		patch_indices.append(inds)
# 		# print(st, ed, observed_tp[inds[0]: inds[-1]+1])

# 		st += args.stride
# 		ed += args.stride

# 	offset = 0
# 	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
# 	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
# 	for b, (record_id, tt, vals, mask) in enumerate(batch):
# 		# tt = tt.to(device)
# 		# vals = vals.to(device)
# 		# mask = mask.to(device)
# 		indices = inverse_indices[offset:offset+len(tt)]
# 		offset += len(tt)
# 		combined_vals[b, indices] = vals
# 		combined_mask[b, indices] = mask

	# if(args.dataset != 'ushcn'):
	# 	combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, 
	# 		att_min = data_min, att_max = data_max)
	# combined_tt = utils.normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)
		
	# data_dict = {
	# 	"data": combined_vals, # (n_batch, T, D)
	# 	"time_steps": combined_tt, # (T, )
	# 	"mask": combined_mask, # (n_batch, T, D)
	# 	}

# 	data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
# 	# print("patchdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())

# 	return data_dict


def patch_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
	Data form as input:
		batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
		batch_vals: (B, M, L_in, D) tensor containing the observed values.
		batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
	Data form to predict:
		flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
		flat_vals: (B, L_out, D) tensor containing the observed values.
		flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	# combined_tt = combined_tt.to(device)
	# print(combined_tt.shape)
	# print(inverse_indices.shape, np.sum([len(ex[1]) for ex in batch]), inverse_indices.max())
	# print(inverse_indices)

	# the number of observed time points 
	n_observed_tp = torch.lt(combined_tt, args.history).sum()
	observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )
	# print(n_observed_tp, len(combined_tt)-n_observed_tp)
	# print(combined_tt[:n_observed_tp])
	# print(combined_tt[n_observed_tp:])

	patch_indices = []
	st, ed = 0, args.patch_size
	for i in range(args.npatch):
		if(i == args.npatch-1):
			inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
		else:
			inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
		patch_indices.append(inds)
		# print(st, ed, observed_tp[inds[0]: inds[-1]+1])

		st += args.stride
		ed += args.stride

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 
	for b, (record_id, tt, vals, mask) in enumerate(batch):
		# tt = tt.to(device)
		# vals = vals.to(device)
		# mask = mask.to(device)
		indices = inverse_indices[offset:offset+len(tt)]
		offset += len(tt)
		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		tmp_n_observed_tp = torch.lt(tt, args.history).sum()
		predicted_tp.append(tt[tmp_n_observed_tp:])
		predicted_data.append(vals[tmp_n_observed_tp:])
		predicted_mask.append(mask[tmp_n_observed_tp:])

	combined_tt = combined_tt[:n_observed_tp]
	combined_vals = combined_vals[:, :n_observed_tp]
	combined_mask = combined_mask[:, :n_observed_tp]
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	if(args.dataset != 'ushcn'):
		combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, 
			att_min = data_min, att_max = data_max)
		predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)

	combined_tt = utils.normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
	# print(predicted_data.sum(), predicted_tp.sum())
		
	data_dict = {
		"data": combined_vals, # (n_batch, T_o, D)
		"time_steps": combined_tt, # (T_o, )
		"mask": combined_mask, # (n_batch, T_o, D)
		"data_to_predict": predicted_data,
		"tp_to_predict": predicted_tp,
		"mask_predicted_data": predicted_mask,
		}

	data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
	# print("patchdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())

	return data_dict


def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	# n_observed_tps = []
	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 

	for b, (record_id, tt, vals, mask) in enumerate(batch):
		n_observed_tp = torch.lt(tt, args.history).sum()
		# n_observed_tps.append(n_observed_tp)
		observed_tp.append(tt[:n_observed_tp])
		observed_data.append(vals[:n_observed_tp])
		observed_mask.append(mask[:n_observed_tp])
		
		predicted_tp.append(tt[n_observed_tp:])
		predicted_data.append(vals[n_observed_tp:])
		predicted_mask.append(mask[n_observed_tp:])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)
	# print(observed_tp.shape, observed_data.shape, observed_mask.shape,\
	#     predicted_tp.shape, predicted_data.shape, predicted_mask.shape)

	if(args.dataset != 'ushcn'):
		observed_data = utils.normalize_masked_data(observed_data, observed_mask, 
			att_min = data_min, att_max = data_max)
		predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)
	
	observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
	# print(predicted_data.sum(), predicted_tp.sum())

	# print(observed_tp.max())
	# print(predicted_tp.max())
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	# print("vecdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())
	
	return data_dict



def variable_time_collate_fn_ODE(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		combined_tt: (T, ) The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)
	# the number of observed time points 
	n_observed_tp = torch.lt(combined_tt, args.history).sum()

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	# combined_labels = None
	# N_labels = 1

	# combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	# combined_labels = combined_labels.to(device = device)
	
	for b, (record_id, tt, vals, mask) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		# if labels is not None:
		# 	labels = labels.to(device)

		indices = inverse_indices[offset:offset+len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		# if labels is not None:
		# 	combined_labels[b] = labels

	if(args.dataset != 'ushcn'):
		combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, 
			att_min = data_min, att_max = data_max)

	# if torch.max(combined_tt) != 0.:
	# 	combined_tt = combined_tt / torch.max(combined_tt)
	# if time_max != 0.:
	# 	combined_tt = combined_tt / time_max
	# else:
	# 	raise Exception("Zero!")
	combined_tt = utils.normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)

	data_dict = {
		"data": combined_vals, # (n_batch, T, D)
		"time_steps": combined_tt, # (T, )
		"mask": combined_mask, # (n_batch, T, D)
		# "labels": combined_labels
		}

	data_dict = utils.split_and_subsample_batch(data_dict, args, n_observed_tp)
	
	return data_dict



def variable_time_collate_fn_CRU(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):

	# n_observed_tps = []
	time_points = []
	obs = []
	mask_obs = [] 

	for b, (record_id, tt, vals, mask) in enumerate(batch):
		n_observed_tp = torch.lt(tt, args.history).sum()
		# n_observed_tps.append(n_observed_tp)
		time_points.append(tt)
		obs.append(vals)
		mask_obs.append(mask)

	time_points = pad_sequence(time_points, batch_first=True).to(dtype=torch.float64)
	obs = pad_sequence(obs, batch_first=True).to(dtype=torch.float64)
	mask_obs = pad_sequence(mask_obs, batch_first=True)
	mask_targets = mask_obs.clone()
	obs_valid = ~torch.all(mask_obs == 0, dim=-1)

	if(args.dataset != 'ushcn'):
		obs = utils.normalize_masked_data(obs, mask_obs, att_min = data_min, att_max = data_max)
	targets = obs.clone()
	
	# observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	# predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {"obs": obs,
			"time_points": time_points,
			"mask_obs": mask_obs,
			"targets": targets,
			"mask_targets": mask_targets,
			"obs_valid": obs_valid,
			}
	
	return data_dict



if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = PhysioNet('../data/physionet', train=False, download=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	print(dataloader.__iter__().next())
