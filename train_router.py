import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import os

# ================= 配置 =================
DATA_DIR = "core/data/data"
SAVE_PATH = "weights/router_best.pth"
INPUT_DIM = 514
HIDDEN_DIM = 256
BATCH_SIZE = 64  # 加大 Batch Size
EPOCHS = 30  # 轮数不用太多，因为过采样了
LR = 0.001


# ========================================

class LiteRouter(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super(LiteRouter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # 🔥 加大 Dropout 防止过拟合
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 开始训练 (强制平衡采样版)...")

    # 1. 加载数据
    x_path = os.path.join(DATA_DIR, "X_router_enhanced.npy")
    y_path = os.path.join(DATA_DIR, "y_router_enhanced.npy")

    if not os.path.exists(x_path):
        print("❌ 数据未找到")
        return

    X = np.load(x_path)
    y = np.load(y_path)

    # 简单的归一化 (对神经网络很重要)
    # 只归一化前 512 维，后面两维 (Entropy/Margin) 保持原样或单独归一化
    # 这里为了简单，暂不处理，依赖 BatchNorm 或数据本身的分布

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    # 2. 🔥🔥🔥 核心：计算采样权重 🔥🔥🔥
    targets = y_tensor.view(-1).long()
    class_count = torch.bincount(targets)
    print(f"📊 样本分布: 负样本(0): {class_count[0]}, 正样本(1): {class_count[1]}")

    # 赋予正样本极高的权重
    weight = 1. / class_count.float()
    samples_weight = weight[targets]

    # 创建采样器
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # DataLoader 加载 sampler (注意：用了 sampler 就不能 shuffle=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 3. 模型定义
    model = LiteRouter().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # 加正则化

    best_recall = 0.0  # 我们只在这个脚本里看重 Recall (召回率)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 训练循环
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 评估 (直接在全集上测，看看能不能把那 3% 找出来)
        model.eval()
        with torch.no_grad():
            full_pred = model(X_tensor.to(device))
            predicted = (full_pred > 0.5).float()
            gt = y_tensor.to(device)

            # 计算 TP, TN, FP, FN
            tp = ((predicted == 1) & (gt == 1)).sum().item()
            fn = ((predicted == 0) & (gt == 1)).sum().item()  # 漏报 (最严重的)
            fp = ((predicted == 1) & (gt == 0)).sum().item()  # 误报

            recall = tp / (tp + fn + 1e-8) * 100
            precision = tp / (tp + fp + 1e-8) * 100

        print(
            f"Epoch {epoch + 1:02d} | Loss: {total_loss / len(train_loader):.4f} | 召回率(Recall): {recall:.2f}% | 精确率: {precision:.2f}%")

        # 我们保存 Recall 最高的模型 (宁可错杀，不可漏报)
        if recall > best_recall and recall > 80.0:  # 至少召回80%才存
            best_recall = recall
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 模型保存 (High Recall)")

    print(f"\n🎉 训练结束。最佳召回率: {best_recall:.2f}%")
    print("现在 Router 应该非常'敏感'了，几乎不会漏掉任何一个难例。")


if __name__ == "__main__":
    train()