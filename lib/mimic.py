import os
import numpy as np
import pandas as pd
import torch

from scipy import special
from torch.utils.data import Dataset

class MIMIC(object):

    def __init__(self, root, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.device = device

        self.process()

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, 'mimic.pt'), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, 'mimic.pt'))

        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]

    def process(self):
        if self._check_exists():
            return
        
        filename = os.path.join(self.raw_folder, 'full_dataset.csv')
        
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing {}...'.format(filename))

        full_data = pd.read_csv(filename, index_col=0)

        patients = []
        value_cols = [c.startswith('Value') for c in full_data.columns]
        value_cols = list(full_data.columns[value_cols])
        mask_cols = [('Mask' + x[5:]) for x in value_cols]
        # print(value_cols)
        # print(mask_cols)
        data_gp = full_data.groupby(level=0) # group by index
        for record_id, data in data_gp:
            tt = torch.tensor(data['Time'].values).to(self.device).float() / 60.
            vals = torch.tensor(data[value_cols].values).to(self.device).float()
            mask = torch.tensor(data[mask_cols].values).to(self.device).float()
            patients.append((record_id, tt, vals, mask))

        torch.save(
            patients,
            os.path.join(self.processed_folder, 'mimic.pt')
        )

        print('Total records:', len(patients))

        print('Done!')

    def _check_exists(self):

        if not os.path.exists(os.path.join(self.processed_folder, 'mimic.pt')):
            return False
        
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

class ITSDataset(Dataset):
    """
    Dataset class for irregular data, originally taken from
    https://github.com/edebrouwer/gru_ode_bayes
    and modified
    """
    def __init__(self, in_df, validation=False, val_options=None):
        """
        Keyword arguments:
        in_df -- patient data (pd.DataFrame)
        validation -- if the constructed dataset is used for validation (bool)
        val_options -- options to specify forecasting time period (dict)
        """
        self.validation = validation
        self.df = in_df

        # Create Dummy covariates and labels if they are not fed.
        num_unique = np.zeros(self.df['ID'].nunique())
        self.init_cov_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'Cov': num_unique})
        self.label_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'label': num_unique})

        if self.validation:
            before_idx = self.df.loc[self.df['Time'] <= val_options['T_val'], 'ID'].unique()
            if val_options.get("T_stop"):
                after_idx = self.df.loc[(self.df['Time'] > val_options['T_val']) & (self.df['Time'] < val_options['T_stop']), 'ID'].unique()
            else:
                after_idx = self.df.loc[self.df['Time'] > val_options['T_val'], 'ID'].unique()

            valid_idx = np.intersect1d(before_idx, after_idx)
            self.df = self.df.loc[self.df['ID'].isin(valid_idx)].copy()
            self.label_df = self.label_df.loc[self.label_df['ID'].isin(valid_idx)].copy()
            self.init_cov_df = self.init_cov_df.loc[self.init_cov_df['ID'].isin(valid_idx)].copy()

        map_dict = dict(zip(self.df.loc[:, 'ID'].unique(), np.arange(self.df.loc[:, 'ID'].nunique())))
        self.map_dict = map_dict
        self.df.loc[:, 'ID'] = self.df.loc[:, 'ID'].map(map_dict)
        self.init_cov_df.loc[:, 'ID'] = self.init_cov_df.loc[:, 'ID'].map(map_dict)
        self.label_df.loc[:, 'ID'] = self.label_df['ID'].map(map_dict)

        assert self.init_cov_df.shape[0] == self.df['ID'].nunique()

        self.variable_num = sum([c.startswith('Value') for c in self.df.columns])
        self.init_cov_dim = self.init_cov_df.shape[1] - 1
        self.init_cov_df = self.init_cov_df.astype(np.float32)
        self.init_cov_df.set_index('ID', inplace=True)
        self.label_df.set_index('ID', inplace=True)
        self.df = self.df.astype(np.float32)

        if self.validation:
            assert val_options is not None, 'Validation set options should be fed'
            self.df_before = self.df.loc[self.df['Time'] <= val_options['T_val']].copy()
            self.df_after = self.df.loc[self.df['Time'] > val_options['T_val']].sort_values('Time').copy()
            if val_options.get("T_stop"):
                self.df_after = self.df_after.loc[self.df_after['Time'] < val_options['T_stop']].sort_values('Time').copy()
            self.df_after = self.df_after.groupby('ID').head(val_options['max_val_samples']).copy()
            self.df = self.df_before  # We remove observations after T_val
            self.df_after.ID = self.df_after.ID.astype(np.int32)
            self.df_after.sort_values('Time', inplace=True)
        else:
            self.df_after = None

        self.length = self.df['ID'].nunique()
        self.df.ID = self.df.ID.astype(np.int32)
        self.df.set_index('ID', inplace=True)
        self.df.sort_values('Time', inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        if len(subset.shape) == 1:
            subset = self.df.loc[[idx]]
        init_covs = self.init_cov_df.loc[idx].values
        tag = self.label_df.loc[idx].astype(np.float32).values
        if self.validation:
            val_samples = self.df_after.loc[self.df_after['ID'] == idx]
        else:
            val_samples = None
        return {'idx': idx, 'y': tag, 'path': subset, 'init_cov': init_covs, 'val_samples': val_samples}


def collate_GOB(batch):
    """
    Collate function used in the DataLoader to format data for GRU-ODE-Bayes,
    taken from https://github.com/edebrouwer/gru_ode_bayes
    """
    df = pd.concat([b['path'] for b in batch], axis=0)
    df.reset_index(inplace=True)
    map_dict = dict(zip(df['ID'].unique(), np.arange(df['ID'].nunique())))
    df['ID'] = df['ID'].map(map_dict)
    df.set_index('ID', inplace=True)
    df.sort_values(by=['ID', 'Time'], inplace=True)

    df_cov = torch.Tensor([b['init_cov'] for b in batch])
    labels = torch.tensor([b['y'] for b in batch])

    times = [df.loc[i].Time.values if isinstance(df.loc[i].Time, pd.Series)
             else np.array([df.loc[i].Time]) for i in df.index.unique()]

    num_observations = [len(x) for x in times]
    value_cols = [c.startswith('Value') for c in df.columns]
    mask_cols = [c.startswith('Mask') for c in df.columns]

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b['val_samples'] for b in batch)
        df_after['ID'] = df_after['ID'].map(map_dict)
        df_after.sort_values(by=['ID', 'Time'], inplace=True)
        df_after.set_index('ID', inplace=True)
        value_cols_val = [c.startswith('Value') for c in df_after.columns]
        mask_cols_val = [c.startswith('Mask') for c in df_after.columns]
        x_val = torch.tensor(df_after.iloc[:, value_cols_val].values)
        m_val = torch.tensor(df_after.iloc[:, mask_cols_val].values)
        times_val = [df_after.loc[i].Time.values if isinstance(df_after.loc[i].Time, pd.Series)
                     else np.array([df_after.loc[i].Time]) for i in df_after.index.unique()]
    else:
        x_val = None
        m_val = None
        times_val = None

    res = dict()
    res['times'] = np.array(times, dtype=object)
    res['num_obs'] = torch.Tensor(num_observations)
    res['X'] = torch.tensor(df.iloc[:, value_cols].values)
    res['M'] = torch.tensor(df.iloc[:, mask_cols].values)
    res['y'] = labels
    res['cov'] = df_cov
    res['X_val'] = x_val
    res['M_val'] = m_val
    res['times_val'] = times_val
    return res

