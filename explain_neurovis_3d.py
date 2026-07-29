"""
NeuroVis-3D: 终极可解释性分析 (Grad-CAM 热图 & 不确定性密度)
完全适配 DST 证据理论满血版架构。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

RESULTS_DIR = r'D:\0临床科研\生物视觉图网络用于分类\NeuroRes_Results_Ultimate'
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================
# 1. 终极版网络架构内置 (防止 import 报错)
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
        return e_fused

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
# 2. Grad-CAM 3D 核心
# ==========================================
class GradCAM3D:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self.save_activation)
                module.register_full_backward_hook(self.save_gradient)
                break
    def save_activation(self, module, input, output): self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0].detach()
    
    def generate(self, x, target_class):
        self.model.zero_grad()
        e_fused = self.model(x)
        # 针对 DST 融合结果求导
        loss = e_fused[0, target_class]
        loss.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros(x.shape[2:])
            
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        if cam.max() > 0: cam = (cam - cam.min()) / (cam.max() + 1e-8)
        else: cam = torch.zeros_like(cam)
        return F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=False).squeeze().cpu().numpy()

def visualize_samples(model, dataset, dataset_name):
    """自动挑选典型样本并生成 M路/P路 关注点对比图"""
    device = next(model.parameters()).device
    model.eval()
    
    best_correct_idx, highest_conf = -1, 0.0
    worst_idx, highest_uncert = -1, 0.0
    
    num_classes = model.num_classes
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("🔍 正在扫描特征样本...")
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)
            label = label.item()
            e_fused = model(img)
            
            S = torch.sum(e_fused + 1.0, dim=1, keepdim=True)
            prob = ((e_fused + 1.0) / S)[0, label].item()
            pred = torch.argmax(e_fused, dim=1).item()
            uncert = (num_classes / S)[0, 0].item()
            
            if pred == label and prob > highest_conf:
                highest_conf, best_correct_idx = prob, i
            if uncert > highest_uncert:
                highest_uncert, worst_idx = uncert, i
                
            if i > 200: break # 只扫前 200 个节省时间

    for desc, idx in [("Confident_Correct", best_correct_idx), ("High_Uncertainty", worst_idx)]:
        if idx == -1: continue
        img_3d, true_label = dataset[idx]
        img_tensor = img_3d.clone().unsqueeze(0).to(device)
        
        # 为了 GradCAM，需要打开 requires_grad
        img_tensor.requires_grad_(True)
        
        e_fused = model(img_tensor)
        S = torch.sum(e_fused + 1.0, dim=1, keepdim=True)
        pred_label = torch.argmax(e_fused, dim=1).item()
        prob = ((e_fused + 1.0) / S)[0, pred_label].item()
        uncert = (num_classes / S)[0, 0].item()
        
        cam_m = GradCAM3D(model, 'm_net.2.conv2').generate(img_tensor, pred_label)
        cam_p = GradCAM3D(model, 'p_net.2.conv2').generate(img_tensor, pred_label)
        
        slice_idx = img_3d.squeeze().shape[0] // 2
        img_slice = img_3d.squeeze()[slice_idx, :, :].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        status = "✅ Correct" if true_label == pred_label else "❌ Wrong"
        fig.suptitle(f'[{status}] True: {true_label} | Pred: {pred_label} | Conf: {prob:.2f} | Uncert: {uncert:.4f}', fontsize=20, fontweight='bold', y=1.05)
        
        axes[0].imshow(img_slice, cmap='gray'); axes[0].set_title('Original Input Slice', fontsize=18); axes[0].axis('off')
        axes[1].imshow(img_slice, cmap='gray'); axes[1].imshow(cam_m[slice_idx], cmap='jet', alpha=0.45)
        axes[1].set_title('M-Pathway Focus (Macro/Low-freq)', fontsize=18, color='darkblue'); axes[1].axis('off')
        axes[2].imshow(img_slice, cmap='gray'); axes[2].imshow(cam_p[slice_idx], cmap='jet', alpha=0.45)
        axes[2].set_title('P-Pathway Focus (Micro/High-freq)', fontsize=18, color='darkred'); axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f'{dataset_name}_{desc}_GradCAM.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"✅ 热图已保存: {save_path}")

def plot_uncertainty_density(model, test_loader, num_classes, dataset_name, device):
    """绘制不确定性分布：证明模型能区分正确与错误的样本"""
    model.eval()
    correct_u, wrong_u = [], []
    
    print("🔍 正在计算不确定性分布...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.numpy()
            e_fused = model(imgs)
            
            S = torch.sum(e_fused + 1.0, dim=1, keepdim=True)
            preds = torch.argmax(e_fused, dim=1).cpu().numpy()
            uncerts = (num_classes / S).squeeze().cpu().numpy()
            
            for i in range(len(labels)):
                if preds[i] == labels[i]: correct_u.append(uncerts[i])
                else: wrong_u.append(uncerts[i])
                
    plt.figure(figsize=(8, 6))
    if len(correct_u) > 0: sns.kdeplot(correct_u, fill=True, color="#2ECC71", alpha=0.5, label=f"Correct (N={len(correct_u)})")
    if len(wrong_u) > 0: sns.kdeplot(wrong_u, fill=True, color="#E74C3C", alpha=0.5, label=f"Wrong (N={len(wrong_u)})")
    
    plt.title(f"Epistemic Uncertainty Density ({dataset_name})", fontsize=16, fontweight='bold')
    plt.xlabel("Dempster-Shafer Uncertainty (u)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(FIGURES_DIR, f'{dataset_name}_Uncertainty_Density.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分布图已保存: {save_path}")

def main():
    print("="*70)
    print("🔥 NeuroVis-3D: Grad-CAM 与不确定性临床价值评估")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['nodulemnist3d', 'fracturemnist3d', 'organmnist3d', 'vesselmnist3d']
    
    for ds in datasets:
        print(f"\n👉 处理数据集: {ds}")
        num_classes = len(INFO[ds]['label'])
        
        weight_path = os.path.join(WEIGHTS_DIR, f'dst_{ds}.pth')
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {ds} (未找到已训练的 dst_{ds}.pth)")
            continue
            
        model = NeuroVis3D_Evidential(num_classes).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
        DataClass = getattr(medmnist, INFO[ds]['python_class'])
        test_dataset = MedMNIST3DDataset(DataClass, split='test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        visualize_samples(model, test_dataset, ds)
        plot_uncertainty_density(model, test_loader, num_classes, ds, device)
        
    print("\n🎉 全部解释性图表生成完毕！请前往 paper_figures 查看！")

if __name__ == "__main__":
    main()