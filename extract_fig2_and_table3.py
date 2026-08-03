"""
NeuroVis-3D: 独立离线图表提取脚本
功能 1: 读取现有的 dst_*.pth 提取高置信度失败案例 (Table III)
功能 2: 锁定随机种子进行“影子训练”，安全重构训练动态曲线 (Fig. 2)
本脚本为纯读取与绘图，绝对不会修改或污染原始定稿权重！
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 目录配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'NeuroRes_Results_Ultimate')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 1. 网络与损失定义 ====================
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
        f_m = self.m_proj(self.pool_m(self.m_net(self.stem_m(m_input))).view(x.size(0), -1))
        f_p = self.p_proj(self.pool_p(self.p_net(self.stem_p(p_input))).view(x.size(0), -1))
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
        return e_fused, e_m, e_p, f_m, f_p

def edl_loss(evidence, target, num_classes, epoch_num, max_epochs=60):
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    y = F.one_hot(target.view(-1).long(), num_classes).float()
    loss_ce = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    annealing = min(1.0, epoch_num / 20.0) 
    loss_kl = annealing * torch.sum(torch.lgamma(kl_alpha) - torch.lgamma(torch.ones_like(kl_alpha)), dim=1, keepdim=True)
    return torch.mean(loss_ce + 0.01 * loss_kl)

class MedMNIST3DDataset(torch.utils.data.Dataset):
    def __init__(self, data_class, split='train'): self.dataset = data_class(split=split, download=True)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        return img.unsqueeze(0) if img.dim()==3 else img, int(label[0])

# ==================== 2. 功能一：提取 Table III 失败案例 ====================
def print_table_iii_failures(dataset_name):
    print(f"\n" + "="*70)
    print(f"📋 正在提取 TABLE III: 高置信度失败案例 ({dataset_name.upper()})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(INFO[dataset_name]['label'])
    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    test_loader = DataLoader(MedMNIST3DDataset(DataClass, 'test'), batch_size=32, shuffle=False)
    
    model = NeuroVis3D_Evidential(num_classes).to(device)
    weight_path = os.path.join(WEIGHTS_DIR, f'dst_{dataset_name}.pth')
    
    if not os.path.exists(weight_path):
        print(f"❌ 找不到权威权重文件: {weight_path}")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    failures = []
    sample_idx = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            e_f, _, _, _, _ = model(imgs)
            S = torch.sum(e_f + 1.0, dim=1, keepdim=True)
            probs = ((e_f + 1.0) / S).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = labels.numpy()
            
            for i in range(len(labels)):
                true_label, pred_label = labels[i], preds[i]
                conf = probs[i, pred_label]
                # 寻找预测错误 且 置信度较高(>0.80) 的样本
                if true_label != pred_label and conf > 0.80:
                    failures.append({
                        'Sample_ID': sample_idx, 'True_Label': true_label, 'Pred_Label': pred_label,
                        'Prob_1': probs[i, 1] if num_classes == 2 else 0, 'Conf': conf
                    })
                sample_idx += 1
                
    failures = sorted(failures, key=lambda x: x['Conf'], reverse=True)
    
    print("-" * 80)
    print(f"| {'Sample ID':<10} | {'True Label':<10} | {'Pred Label':<10} | {'Prob (Target 1)':<15} | {'Confidence':<12} |")
    print("-" * 80)
    for f in failures[:10]:
        print(f"| {f['Sample_ID']:<10} | {f['True_Label']:<10} | {f['Pred_Label']:<10} | {f['Prob_1']:<15.4f} | {f['Conf']:<12.4f} |")
    print("-" * 80)

# ==================== 3. 功能二：影子训练重构 Fig. 2 ====================
def evaluate_auc(model, loader, device, is_neuro):
    model.eval()
    probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if is_neuro:
                e_f, _, _, _, _ = model(imgs.to(device))
                probs.append(((e_f+1)/torch.sum(e_f+1, dim=1, keepdim=True)).cpu().numpy()[:, 1])
            else:
                probs.append(torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()[:, 1])
            all_labels.append(labels.numpy())
    return roc_auc_score(np.concatenate(all_labels), np.concatenate(probs))

def generate_fig2_curves(dataset_name, epochs=60):
    set_seed(42) # 严格锁定种子，确保完美复刻
    print(f"\n" + "="*70)
    print(f"📈 正在进行影子训练重构 Fig. 2 曲线图 ({dataset_name.upper()})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(INFO[dataset_name]['label'])
    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    
    train_loader = DataLoader(MedMNIST3DDataset(DataClass, 'train'), batch_size=32, shuffle=True)
    test_loader = DataLoader(MedMNIST3DDataset(DataClass, 'test'), batch_size=32, shuffle=False)
    
    base_model = VanillaResNet3D(num_classes).to(device)
    our_model = NeuroVis3D_Evidential(num_classes).to(device)
    opt_base = torch.optim.AdamW(base_model.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_our = torch.optim.AdamW(our_model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch_our = torch.optim.lr_scheduler.CosineAnnealingLR(opt_our, T_max=epochs)
    
    history = {'base_loss': [], 'base_auc': [], 'our_loss': [], 'our_auc': [], 'm_ev': [], 'p_ev': []}
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"  Reconstructing Curves"):
        # Base
        base_model.train()
        ep_base_loss = 0
        for imgs, labels in train_loader:
            opt_base.zero_grad()
            loss = nn.CrossEntropyLoss()(base_model(imgs.to(device)), labels.to(device).long())
            loss.backward(); opt_base.step()
            ep_base_loss += loss.item()
        history['base_loss'].append(ep_base_loss / len(train_loader))
        history['base_auc'].append(evaluate_auc(base_model, test_loader, device, is_neuro=False))
        
        # Ours
        our_model.train()
        ep_our_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_our.zero_grad()
            e_f, e_m, e_p, f_m, f_p = our_model(imgs)
            loss_our = edl_loss(e_f, labels, num_classes, epoch, epochs) + 0.5*edl_loss(e_m, labels, num_classes, epoch, epochs) + 0.5*edl_loss(e_p, labels, num_classes, epoch, epochs) + 0.1*torch.mean(torch.abs(F.cosine_similarity(f_m, f_p, dim=1)))
            loss_our.backward(); torch.nn.utils.clip_grad_norm_(our_model.parameters(), 1.0); opt_our.step()
            ep_our_loss += loss_our.item()
        sch_our.step()
        
        history['our_loss'].append(ep_our_loss / len(train_loader))
        history['our_auc'].append(evaluate_auc(our_model, test_loader, device, is_neuro=True))
        
        our_model.eval()
        m_ev_avg, p_ev_avg = [], []
        with torch.no_grad():
            for imgs, _ in test_loader:
                _, e_m, e_p, _, _ = our_model(imgs.to(device))
                m_ev_avg.append(torch.mean(torch.sum(e_m, dim=1)).item())
                p_ev_avg.append(torch.mean(torch.sum(e_p, dim=1)).item())
        history['m_ev'].append(np.mean(m_ev_avg))
        history['p_ev'].append(np.mean(p_ev_avg))

    # 绘图 (严格去除独立通路曲线，只展示核心指标)
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['base_loss'], label='ResNet Loss', color='#3498db', linewidth=2)
    axes[0].plot(history['our_loss'], label='NeuroVis Loss', color='#e74c3c', linewidth=2)
    axes[0].set_title(f'Training Loss ({dataset_name})', fontweight='bold'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[1].plot(history['base_auc'], label='ResNet AUC', color='#3498db', linewidth=2)
    axes[1].plot(history['our_auc'], label='NeuroVis Fusion AUC', color='#e74c3c', linewidth=2)
    axes[1].set_title(f'Validation AUC ({dataset_name})', fontweight='bold'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[2].plot(history['m_ev'], label='M-Pathway Evidence', color='#2ecc71', linewidth=2)
    axes[2].plot(history['p_ev'], label='P-Pathway Evidence', color='#f39c12', linewidth=2)
    axes[2].set_title(f'Average Evidence Contribution ({dataset_name})', fontweight='bold'); axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'Fig2_{dataset_name}_Training_Curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 成功生成高清训练曲线图，保存至: {save_path}")

def main():
    print("="*80)
    print("🎨 NeuroVis-3D 离线图表提取工具 (保证不污染原始训练结果)")
    print("="*80)
    
    # 默认只针对 Nodule 生成图表（可自行修改数组添加 'organmnist3d', 'vesselmnist3d' 等）
    datasets_to_process = ['nodulemnist3d'] 
    
    for ds in datasets_to_process:
        print_table_iii_failures(ds)     # 只读权重，提取失败案例
        generate_fig2_curves(ds)         # 影子训练，画曲线图
        
    print("\n🎉 全部附加图表提取完毕！")

if __name__ == "__main__":
    main()
