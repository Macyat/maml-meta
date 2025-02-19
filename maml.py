import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SpectralRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(590, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr  # 直接存储学习率
        self.outer_optim = optim.Adam(self.model.parameters(), lr=outer_lr)

    def adapt(self, support_X, support_y, num_steps=5):
        """内部循环：手动梯度下降，无需优化器step()"""
        fast_weights = list(self.model.parameters())
        for _ in range(num_steps):
            pred = self.model(support_X)
            loss = F.mse_loss(pred, support_y)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            # 使用self.inner_lr而非param_groups
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        return fast_weights

    def evaluate(self, query_X, query_y, fast_weights):
        """外部循环：更新初始参数"""
        self.outer_optim.zero_grad()
        pred = self.model(query_X)
        loss = F.mse_loss(pred, query_y)
        loss.backward()
        self.outer_optim.step()
        return loss.item()


# 使用示例
model = SpectralRegressor()
maml = MAML(model, inner_lr=0.01)
