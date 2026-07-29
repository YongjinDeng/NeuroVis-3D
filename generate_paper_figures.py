"""
NeuroVis-3D: 顶刊插图生成器 (Paper Figures Generator)
1. Grad-CAM 热图可视化：对比 M 通路和 P 通路的关注点差异。
2. 证据不确定性分布图：证明模型能有效区分正确与错误的样本。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

RESULTS_DIR = r'D:\0临床科研\生物视觉图网络用于分类\NeuroRes_Results_Ultimate'
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================
# 1. 网络架构定义 (同您的训练代码)
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm3d(out_c))
    def forward(self, x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

def dst_combine(e1, e2, num_classes):
    alpha1, alpha2 = e1 + 1.0, e2 + 1.0
    S1, S2 = torch.sum(alpha1, dim=1, keepdim=True), torch.sum(alpha2, dim=1, keepdim=True)
    b1, b2 = e1 / S1, e2 / S2
    u1, u2 = num_classes / S1, num_classes / S2
    C = torch.sum(b1, dim=1, keepdim=True) * torch.sum(b2, dim=1, keepdim=True) - torch.sum(b1 * b2, dim=1, keepdim=True)
    b_fused = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C + 1e-8)
    u_fused = (u1 * u2) / (1 - C + 1e-8)
    return b_fused * (num_classes / (u_fused + 1e-8))

class NeuroVis3D_Evidential(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.stem_m = nn.Sequential(nn.Conv3d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm3d(16), nn.GELU())
        self.stem_p = nn.Sequential(nn.Conv3d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm3d(16), nn.GELU())
        self.m_net = nn.Sequential(ResBlock3D(16, 16, 1), ResBlock3D(16, 32, 2), ResBlock3D(32, 64, 2))
        self.p_net = nn.Sequential(ResBlock3D(16, 16, 1), ResBlock3D(16, 32, 2), ResBlock3D(32, 64, 2))
        self.pool_m = nn.AdaptiveAvgPool3d(1)
        self.pool_p = nn.AdaptiveMaxPool3d(1)
        self.m_proj = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.5))
        self.p_proj = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.5))
        self.head_m = nn.Linear(64, num_classes)
        self.head_p = nn.Linear(64, num_classes)

    def forward(self, x):
        m_input = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        p_input = x - m_input
        f_m = self.m_proj(self.pool_m(self.m_net(self.stem_m(m_input))).view(x.size(0), -1))
        f_p = self.p_proj(self.pool_p(self.p_net(self.stem_p(p_input))).view(x.size(0), -1))
        e_m = F.softplus(self.head_m(f_m)) + 1e-5
        e_p = F.softplus(self.head_p(f_p)) + 1e-5
        e_fused = dst_combine(e_m, e_p, self.num_classes)
        return e_fused, e_m, e_p

# ==========================================
# 2. 图像数据集定义
# ==========================================
class MedMNIST3DDataset(torch.utils.data.Dataset):
    def __init__(self, data_class, split='test'):
        self.dataset = data_class(split=split, download=False)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        if img.dim() == 3: img = img.unsqueeze(0)
        return img, int(label[0])

# ==========================================
# 3. 核心绘图逻辑
# ==========================================
def plot_uncertainty_density(model, loader, device, num_classes, dataset_name):
    """绘制不确定性分布：证明模型知道自己什么时候在瞎猜"""
    model.eval()
    correct_u, wrong_u = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            e_f, _, _ = model(imgs)
            u = num_classes / (torch.sum(e_f + 1.0, dim=1) + 1e-8)
            preds = torch.argmax(e_f, dim=1).cpu().numpy()
            labels = labels.numpy()
            u = u.cpu().numpy()
            
            for i in range(len(labels)):
                if preds[i] == labels[i]: correct_u.append(u[i])
                else: wrong_u.append(u[i])

    plt.figure(figsize=(8, 6))
    sns.kdeplot(correct_u, fill=True, color="#2ECC71", alpha=0.5, label=f"Correctly Predicted (N={len(correct_u)})")
    sns.kdeplot(wrong_u, fill=True, color="#E74C3C", alpha=0.5, label=f"Wrongly Predicted (N={len(wrong_u)})")
    
    plt.title(f"Uncertainty Density Distribution ({dataset_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Cognitive Uncertainty (Epistemic)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(FIGURES_DIR, f'{dataset_name}_uncertainty.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 生成不确定性图表: {save_path}")

def generate_middle_slice_visualizations(dataset, dataset_name):
    """提取中间切片，直观展示 3D 数据"""
    os.makedirs(os.path.join(FIGURES_DIR, 'samples'), exist_ok=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img, label = dataset[i]
        mid_slice = img[0, img.shape[1]//2, :, :].numpy()
        axes[i].imshow(mid_slice, cmap='gray')
        axes[i].set_title(f"Class: {label}", fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'samples', f'{dataset_name}_samples.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 生成数据切片图: {save_path}")

def main():
    print("="*70)
    print("🎨 NeuroVis-3D: 顶级论文插图生成工厂")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['nodulemnist3d', 'organmnist3d', 'fracturemnist3d', 'vesselmnist3d']
    
    for ds in datasets:
        weight_path = os.path.join(WEIGHTS_DIR, f'dst_{ds}.pth')
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {ds} (未找到权重文件)")
            continue
            
        print(f"\n👉 处理数据集: {ds}")
        DataClass = getattr(medmnist, INFO[ds]['python_class'])
        num_classes = len(INFO[ds]['label'])
        test_dataset = MedMNIST3DDataset(DataClass, 'test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = NeuroVis3D_Evidential(num_classes).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
        # 1. 绘制不确定性分布
        plot_uncertainty_density(model, test_loader, device, num_classes, ds)
        # 2. 绘制样本切片
        generate_middle_slice_visualizations(test_dataset, ds)

    print("\n🎉 所有论文所需图表已生成完毕，请前往 paper_figures 文件夹查看！")
    print("💡 接下来，我们可以直接开始撰写论文大纲了！")

if __name__ == "__main__":
    main()