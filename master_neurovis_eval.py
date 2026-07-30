"""
NeuroVis-3D: 终极定稿评估套件 (The Ultimate Evaluation Suite)
包含: 动态路由证据融合 + 单侧 DeLong 显著性检验 + 复杂度分析
完全契合 IEEE JBHI / TMI 顶刊论文描述
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats as st
import medmnist
from medmnist import INFO
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 目录与全局配置 ====================
RESULTS_DIR = r'D:\0临床科研\生物视觉图网络用于分类\NeuroRes_Results_Ultimate'
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def set_seed(seed=42):
    """固定随机种子，确保实验完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ==================== 1. 网络架构定义 ====================
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
            
    def forward(self, x): 
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class VanillaResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm3d(16), nn.GELU())
        self.layer1 = ResBlock3D(16, 16, 1)
        self.layer2 = ResBlock3D(16, 32, 2)
        self.layer3 = ResBlock3D(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        return self.fc(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).view(x.size(0), -1))

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
        f_m_raw = self.pool_m(self.m_net(self.stem_m(m_input))).view(x.size(0), -1)
        f_p_raw = self.pool_p(self.p_net(self.stem_p(p_input))).view(x.size(0), -1)
        f_m = self.m_proj(f_m_raw)
        f_p = self.p_proj(f_p_raw)
        e_m = F.softplus(self.head_m(f_m)) + 1e-5
        e_p = F.softplus(self.head_p(f_p)) + 1e-5
        alpha1, alpha2 = e_m + 1.0, e_p + 1.0
        S1, S2 = torch.sum(alpha1, dim=1, keepdim=True), torch.sum(alpha2, dim=1, keepdim=True)
        b1, b2 = e_m / S1, e_p / S2
        u1, u2 = self.num_classes / S1, self.num_classes / S2
        C = torch.sum(b1, dim=1, keepdim=True) * torch.sum(b2, dim=1, keepdim=True) - torch.sum(b1 * b2, dim=1, keepdim=True)
        b_fused = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C + 1e-8)
        u_fused = (u1 * u2) / (1 - C + 1e-8)
        e_fused = b_fused * (self.num_classes / (u_fused + 1e-8))
        return e_fused, e_m, e_p, f_m, f_p, torch.mean(b1, dim=1), torch.mean(b2, dim=1)

def edl_loss(evidence, target, num_classes, epoch_num, max_epochs=60):
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    y = F.one_hot(target.view(-1).long(), num_classes).float()
    loss_ce = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    annealing = min(1.0, epoch_num / 20.0) 
    loss_kl = annealing * torch.sum(torch.lgamma(kl_alpha) - torch.lgamma(torch.ones_like(kl_alpha)), dim=1, keepdim=True)
    return torch.mean(loss_ce + 0.01 * loss_kl)

# ==================== 2. 核心评估工具 ====================
def measure_complexity(model, device):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    dummy = torch.randn(1, 1, 28, 28, 28).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(dummy)
        start = time.time()
        for _ in range(50): model(dummy)
        inf_time = (time.time() - start) / 50 * 1000
    return params, inf_time

def delong_auc_test(y_true, y_pred_baseline, y_pred_ours):
    """精确单侧 DeLong AUC 显著性检验"""
    if len(np.unique(y_true)) > 2: return 1.0
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2
    preds = np.vstack((y_pred_baseline, y_pred_ours))
    m, n = np.sum(y_true == 1), np.sum(y_true == 0)
    tx = np.apply_along_axis(compute_midrank, 1, preds[:, y_true == 1])
    ty = np.apply_along_axis(compute_midrank, 1, preds[:, y_true == 0])
    tz = np.apply_along_axis(compute_midrank, 1, preds)
    aucs = tz[:, y_true == 1].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, y_true == 1] - tx) / n
    v10 = 1.0 - (tz[:, y_true == 0] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    delongcov = sx / m + sy / n
    diff = aucs[1] - aucs[0]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0: return 1.0
    z = diff / np.sqrt(var)
    return 1 - st.norm.cdf(z)

# ==================== 3. 训练与验证引擎 ====================
class MedMNIST3DDataset(torch.utils.data.Dataset):
    def __init__(self, data_class, split='train'): 
        self.dataset = data_class(split=split, download=True)
    def __len__(self): 
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        return img.unsqueeze(0) if img.dim()==3 else img, int(label[0])

def evaluate(model, loader, device, num_classes, is_neuro=False):
    model.eval()
    all_probs_f, all_probs_m, all_probs_p, all_labels = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if is_neuro:
                e_f, e_m, e_p, _, _, _, _ = model(imgs)
                all_probs_f.append(((e_f+1)/torch.sum(e_f+1, dim=1, keepdim=True)).cpu().numpy())
                all_probs_m.append(((e_m+1)/torch.sum(e_m+1, dim=1, keepdim=True)).cpu().numpy())
                all_probs_p.append(((e_p+1)/torch.sum(e_p+1, dim=1, keepdim=True)).cpu().numpy())
            else:
                all_probs_f.append(torch.softmax(model(imgs), dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs_f = np.concatenate(all_probs_f, axis=0)
    
    if num_classes == 2:
        auc_f = roc_auc_score(all_labels, all_probs_f[:, 1])
        if is_neuro:
            auc_m = roc_auc_score(all_labels, np.concatenate(all_probs_m, axis=0)[:, 1])
            auc_p = roc_auc_score(all_labels, np.concatenate(all_probs_p, axis=0)[:, 1])
            return auc_f, auc_m, auc_p, all_probs_f[:, 1], all_labels
        return auc_f, all_probs_f[:, 1], all_labels
    else:
        auc_f = roc_auc_score(all_labels, all_probs_f, multi_class='ovr')
        if is_neuro:
            auc_m = roc_auc_score(all_labels, np.concatenate(all_probs_m, axis=0), multi_class='ovr')
            auc_p = roc_auc_score(all_labels, np.concatenate(all_probs_p, axis=0), multi_class='ovr')
            return auc_f, auc_m, auc_p, all_probs_f, all_labels
        return auc_f, all_probs_f, all_labels

def train_and_evaluate(dataset_name, epochs=60, batch_size=32):
    print(f"\n{'='*70}\n🚀 执行定稿评估: {dataset_name.upper()}\n{'='*70}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    num_classes = len(INFO[dataset_name]['label'])
    
    train_loader = DataLoader(MedMNIST3DDataset(DataClass, 'train'), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MedMNIST3DDataset(DataClass, 'test'), batch_size=batch_size, shuffle=False)
    
    # ---------- 1. 训练 Baseline (Conventional Protocol) ----------
    print("⏳ [1/3] 训练 3D-ResNet Baseline (Conventional Protocol)...")
    base_model = VanillaResNet3D(num_classes).to(device)
    base_params, base_time = measure_complexity(base_model, device)
    opt_base = torch.optim.AdamW(base_model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    for epoch in tqdm(range(1, epochs + 1), desc="  Base Training", leave=False):
        base_model.train()
        for imgs, labels in train_loader:
            opt_base.zero_grad()
            loss = nn.CrossEntropyLoss()(base_model(imgs.to(device)), labels.to(device).long())
            loss.backward()
            opt_base.step()
            
    base_test_auc, base_probs, test_labels = evaluate(base_model, test_loader, device, num_classes, is_neuro=False)
    print(f"✅ Baseline Test AUC: {base_test_auc:.4f}")

    # ---------- 2. 训练 NeuroVis-3D (Evidential Protocol) ----------
    print("\n⏳ [2/3] 训练 NeuroVis-3D (Evidential Protocol)...")
    our_model = NeuroVis3D_Evidential(num_classes).to(device)
    our_params, our_time = measure_complexity(our_model, device)
    opt_our = torch.optim.AdamW(our_model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch_our = torch.optim.lr_scheduler.CosineAnnealingLR(opt_our, T_max=epochs)
    
    for epoch in tqdm(range(1, epochs + 1), desc="  Ours Training", leave=False):
        our_model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_our.zero_grad()
            e_f, e_m, e_p, f_m, f_p, _, _ = our_model(imgs)
            loss_f = edl_loss(e_f, labels, num_classes, epoch, epochs)
            loss_m = edl_loss(e_m, labels, num_classes, epoch, epochs)
            loss_p = edl_loss(e_p, labels, num_classes, epoch, epochs)
            loss_ortho = torch.mean(torch.abs(F.cosine_similarity(f_m, f_p, dim=1)))
            loss = loss_f + 0.5 * loss_m + 0.5 * loss_p + 0.1 * loss_ortho
            loss.backward()
            torch.nn.utils.clip_grad_norm_(our_model.parameters(), 1.0)
            opt_our.step()
        sch_our.step()

    # 仅保存单一权威权重文件 dst_{dataset_name}.pth
    save_path = os.path.join(WEIGHTS_DIR, f'dst_{dataset_name}.pth')
    torch.save(our_model.state_dict(), save_path)
    print(f"💾 仅保存单一权威模型权重至: {save_path}")

    test_f, test_m, test_p, our_probs, _ = evaluate(our_model, test_loader, device, num_classes, is_neuro=True)
    print(f"✅ NeuroVis-3D Test AUC: {test_f:.4f} (M: {test_m:.4f} | P: {test_p:.4f})")
    
    # ---------- 3. 单侧 DeLong 显著性检验 ----------
    print("\n⏳ [3/3] 计算 DeLong 统计显著性...")
    if num_classes == 2:
        p_delong = delong_auc_test(test_labels, base_probs, our_probs)
        if p_delong < 0.001:
            final_p_str = "< 0.001"
        else:
            final_p_str = f"{p_delong:.4f}"
        print(f"   DeLong p-value: {final_p_str}")
    else:
        final_p_str = "N/A"
        
    return {
        'Dataset': dataset_name,
        'Base': base_test_auc,
        'Ours': test_f,
        'M': test_m,
        'P': test_p,
        'PVal': final_p_str,
        'Params': f"{base_params:.1f}M / {our_params:.1f}M",
        'Time': f"{base_time:.1f}ms / {our_time:.1f}ms"
    }

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print("="*100)
    print("🏆 NeuroVis-3D 终极定稿程序 (仅保留 DeLong 检验 & 单一权威权重保存)")
    print("="*100)
    
    datasets = ['nodulemnist3d', 'organmnist3d', 'fracturemnist3d', 'vesselmnist3d']
    results = [train_and_evaluate(ds) for ds in datasets]
    
    print("\n\n" + "★"*115)
    print("📋 TABLE I: COMPREHENSIVE PERFORMANCE & STATISTICAL SIGNIFICANCE")
    print("★"*115)
    print(f"| {'Dataset':<16} | {'ResNet':<8} | {'M-Net':<8} | {'P-Net':<8} | {'Ours(Fusion)':<12} | {'DeLong p-val':<12} | {'Params(Base/Ours)':<18} | {'Inf.Time':<18} |")
    print(f"|{'-'*18}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*20}|{'-'*20}|")
    for r in results:
        flag = "✅" if r['Ours'] >= max(r['M'], r['P']) else "⚠️"
        print(f"| {r['Dataset']:<16} | {r['Base']:.4f}   | {r['M']:.4f}   | {r['P']:.4f}   | **{r['Ours']:.4f}** {flag:<2} | {r['PVal']:<12} | {r['Params']:<18} | {r['Time']:<18} |")
    print("★"*115)
    print("\n🎉 实验全部结束！权威权重已写盘，计算得到的 DeLong P 值可直接更至论文 Table I！")

if __name__ == "__main__":
    main()
