import os

import pandas as pd
import torch
from maml import MAML
from meta_task_dataset import MetaTaskDataset
import torch.nn.functional as F


def evaluate_on_new_task_spxy(model, task_folder, task_id, support_ratio=0.2):
    # 加载新任务数据
    task_csv_path = os.path.join(task_folder, f"{task_id}.csv")
    df = pd.read_csv(task_csv_path)
    X_all = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y_all = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

    # 使用SPXY选择支撑集
    selected_indices = MetaTaskDataset.spxy_split(
        X_all.numpy(), y_all.numpy(), n_support=int(len(X_all) * support_ratio)
    )
    support_X = X_all[selected_indices]
    support_y = y_all[selected_indices]
    query_X = X_all[[i for i in range(len(X_all)) if i not in selected_indices]]
    query_y = y_all[[i for i in range(len(y_all)) if i not in selected_indices]]

    # 快速适应
    maml = MAML(model)
    fast_weights = maml.adapt(support_X, support_y, num_steps=5)

    # 评估
    _, pred = maml.evaluate(query_X, query_y, fast_weights)
    rmse = torch.sqrt(F.mse_loss(pred, query_y))

    print(f"Test RMSE: {rmse:.2f}")
    return pred
