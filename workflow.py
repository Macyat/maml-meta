from torch.utils.data import DataLoader
import torch
from maml import MAML, SpectralRegressor
from meta_task_dataset import MetaTaskDataset
from evaluate import evaluate_on_new_task_spxy


def train_maml(config):
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralRegressor().to(device)
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001)

    # 加载数据
    train_tasks = MetaTaskDataset(
        task_folder=config["task_folder"],
        task_ids=config["train_ids"],  # 训练任务
        support_ratio=30,
        mode="train",
    )
    train_loader = DataLoader(train_tasks, batch_size=None, shuffle=True)

    # 训练循环
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for task_data in train_loader:
            # 移至设备
            support_X, support_y = task_data["support"]
            query_X, query_y = task_data["query"]
            support_X, support_y = support_X.to(device), support_y.to(device)
            query_X, query_y = query_X.to(device), query_y.to(device)

            # 内部适应
            fast_weights = maml.adapt(support_X, support_y, num_steps=5)

            # 外部更新
            loss, _ = maml.evaluate(query_X, query_y, fast_weights)
            epoch_loss += loss

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")

    return model


# configs setup
config = {
    "epochs": 100,
    "task_folder": "data",
    "support_ratio": 0.2,
    "query_ratio": 0.8,
    "train_ids": ["task1", "task2"],
    "test_ids": "task3",
}  # 假设每个任务300样本

# train and evaluate
model = train_maml(config)
evaluate_on_new_task_spxy(model, config["task_folder"], config["test_ids"])
