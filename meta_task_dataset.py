import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MetaTaskDataset(Dataset):
    def __init__(self, task_folder, task_ids, support_ratio=0.2, mode="train"):
        """
        :param task_folder: where .csv files (raw data for each site locates)
        :param task_ids: a list of task ids (e.g. ['task1', 'task2'])
        :param support_size: size of support set
        :param mode: 'train'（random sampling）或 'test'（sampling with SPXY）
        """
        self.tasks = []
        self.support_ratio = support_ratio
        self.mode = mode

        # 加载所有任务数据
        for task_id in task_ids:
            csv_path = os.path.join(task_folder, f"{task_id}.csv")
            df = pd.read_csv(csv_path)
            X = df.iloc[:, :-1].values.astype(np.float32)  # (n_samples, 590)
            y = df.iloc[:, -1].values.astype(np.float32)  # (n_samples,)
            self.tasks.append({"X": X, "y": y})

    def __len__(self):
        return len(self.tasks)  # 任务数量

    def __getitem__(self, idx):
        """返回一个任务的支撑集和查询集"""
        task = self.tasks[idx]
        X, y = task["X"], task["y"]

        if self.mode == "train":
            # 训练模式：随机划分
            indices = np.random.permutation(len(X))
        elif self.mode == "test":
            # 测试模式：SPXY算法
            indices = self.spxy_split(X, y, int(len(X) * self.support_ratio))
        else:
            raise ValueError("mode must be 'train' or 'test'")

        support_indices = indices[: int(len(X) * self.support_ratio)]
        query_indices = indices[int(len(X) * self.support_ratio) :]

        # 转换为Tensor
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        return {
            "support": (X_tensor[support_indices], y_tensor[support_indices]),
            "query": (X_tensor[query_indices], y_tensor[query_indices]),
        }

    @staticmethod
    def spxy_split(X, y, n_support, sklearn=None):
        """SPXY算法实现（优化版）"""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1))

        # 加速距离计算
        D_x = np.sqrt(((X_scaled[:, None] - X_scaled) ** 2).sum(axis=2))
        D_y = np.sqrt((y_scaled[:, None] - y_scaled) ** 2)
        D = D_x + D_y

        selected = []
        # 初始选择距离最远的两个样本
        max_pair = np.unravel_index(np.argmax(D), D.shape)
        selected.extend(max_pair)

        # 逐步添加剩余样本
        for _ in range(2, n_support):
            candidates = np.setdiff1d(np.arange(len(X)), selected)
            min_dists = np.min(D[candidates][:, selected], axis=1)
            best_idx = candidates[np.argmax(min_dists)]
            selected.append(best_idx)

        return np.array(selected)


class Normalizer:
    """任务级别的标准化"""

    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8

    def transform(self, X, y):
        X_norm = (X - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        return X_norm, y_norm

    def inverse_transform_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean
