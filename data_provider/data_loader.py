import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='custom.csv',
                 target='power', scale=True, timeenc=1, freq='t', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_val = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
