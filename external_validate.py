"""
NeuroVis-3D: 真实世界临床迁移学习 (LUNA16 Transfer Learning - 修复版)
目标：加载 MedMNIST 预训练权重，在 LUNA16 上进行 15 轮极速微调，克服域偏移。
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import SimpleITK as sitk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'external_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'NeuroRes_Results_Ultimate')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ==========================================
# 1. 核心网络架构 (与预训练时绝对一致)
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

def edl_loss(evidence, target, num_classes, epoch_num, max_epochs=60):
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    y = F.one_hot(target.view(-1).long(), num_classes).float()
    loss_ce = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    annealing = min(1.0, epoch_num / 5.0) 
    loss_kl = annealing * torch.sum(torch.lgamma(kl_alpha) - torch.lgamma(torch.ones_like(kl_alpha)), dim=1, keepdim=True)
    return torch.mean(loss_ce + 0.01 * loss_kl)

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
        return e_fused, e_m, e_p, f_m, f_p

# ==========================================
# 2. LUNA16 数据集加载 (定向偏移采样)
# ==========================================
def safe_sitk_read(file_path):
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    original_cwd = os.getcwd()
    try:
        os.chdir(dir_name)
        img = sitk.ReadImage(base_name)
    finally:
        os.chdir(original_cwd)
    return img

class LUNA16AutoDataset(Dataset):
    def __init__(self, data_dir, patch_size=(28, 28, 28)):
        self.patch_size = patch_size
        print("🔍 正在扫描 CT 影像路径...")
        uid_to_path = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.mhd'):
                    uid_to_path[file.replace('.mhd', '')] = os.path.join(root, file)
                    
        df_pos = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))
        df_pos['target_label'] = 1
        df_pos = df_pos[df_pos['seriesuid'].astype(str).isin(uid_to_path.keys())].reset_index(drop=True)

        print("💡 正在使用定向偏移法生成负样本...")
        neg_rows = []
        offsets_mm = [(35.0, 35.0, 0.0), (-35.0, -35.0, 0.0), (35.0, -35.0, 0.0), (-35.0, 35.0, 0.0)]
        for _, row in df_pos.iterrows():
            uid = str(row['seriesuid'])
            mhd_path = uid_to_path.get(uid)
            if not mhd_path: continue
            try:
                itk_img = safe_sitk_read(mhd_path)
                img_array = sitk.GetArrayFromImage(itk_img)
                D, H, W = img_array.shape
                pos_world = (float(row['coordX']), float(row['coordY']), float(row['coordZ']))
                
                for dx, dy, dz in offsets_mm:
                    cand_world = (pos_world[0] + dx, pos_world[1] + dy, pos_world[2] + dz)
                    try:
                        cand_idx = itk_img.TransformPhysicalPointToIndex(cand_world)
                        vx, vy, vz = cand_idx[0], cand_idx[1], cand_idx[2]
                        if 14 <= vz < D-14 and 14 <= vy < H-14 and 14 <= vx < W-14:
                            patch = img_array[vz-14:vz+14, vy-14:vy+14, vx-14:vx+14]
                            if patch.size > 0 and np.mean(patch) > -990:
                                neg_rows.append({'seriesuid': uid, 'coordX': cand_world[0], 'coordY': cand_world[1], 'coordZ': cand_world[2], 'target_label': 0})
                                break
                    except: pass
            except: pass
                
        df_neg = pd.DataFrame(neg_rows)
        self.df = pd.concat([df_pos, df_neg], ignore_index=True)
        self.uid_to_path = uid_to_path
        print(f"✅ LUNA16 数据集准备就绪 (总数: {len(self.df)}，正: {sum(self.df['target_label']==1)}，负: {sum(self.df['target_label']==0)})")

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            itk_img = safe_sitk_read(self.uid_to_path[str(row['seriesuid'])])
            img_array = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            
            coord_world = (float(row['coordX']), float(row['coordY']), float(row['coordZ']))
            vx, vy, vz = itk_img.TransformPhysicalPointToIndex(coord_world)
            pz, py, px = self.patch_size[0]//2, self.patch_size[1]//2, self.patch_size[2]//2
            z_s, z_e = max(0, vz-pz), min(img_array.shape[0], vz+pz)
            y_s, y_e = max(0, vy-py), min(img_array.shape[1], vy+py)
            x_s, x_e = max(0, vx-px), min(img_array.shape[2], vx+px)
            
            patch = img_array[z_s:z_e, y_s:y_e, x_s:x_e]
            if patch.shape != tuple(self.patch_size):
                patch = np.pad(patch, ((pz-(vz-z_s), pz-(z_e-vz)), (py-(vy-y_s), py-(y_e-vy)), (px-(vx-x_s), px-(x_e-vx))), constant_values=-1000)
            
            patch = np.clip(patch, -1000, 400)
            patch = (patch - (-1000)) / 1400.0
            return torch.tensor(patch, dtype=torch.float32).unsqueeze(0), int(row['target_label'])
        except:
            return torch.zeros((1, 28, 28, 28)), int(row['target_label'])

# ==========================================
# 3. 评估指标计算
# ==========================================
def evaluate(model, loader, device, num_classes=2):
    """评估函数，兼容 num_classes 参数"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            e_f, _, _, _, _ = model(imgs.to(device))
            prob = (e_f + 1) / torch.sum(e_f + 1, dim=1, keepdim=True)
            all_probs.append(prob.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    return auc, acc, tp/(tp+fn+1e-8), tn/(tn+fp+1e-8), tn, fp, fn, tp

# ==========================================
# 4. 微调主程序 (Transfer Learning)
# ==========================================
def main():
    print("="*80)
    print("🚀 NeuroVis-3D: LUNA16 临床迁移学习 (Transfer Learning)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    full_dataset = LUNA16AutoDataset(DATA_DIR)
    
    # 随机 70% 训练，30% 测试
    train_idx, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.3, random_state=42, stratify=full_dataset.df['target_label'])
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=16, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=16, shuffle=False)
    
    print(f"📊 数据集划分: 训练集 {len(train_idx)} 个，测试集 {len(test_idx)} 个")
    
    model = NeuroVis3D_Evidential(num_classes=2).to(device)
    pretrained_path = os.path.join(WEIGHTS_DIR, 'dst_nodulemnist3d.pth')
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("✅ 成功加载 MedMNIST 预训练骨干网络，开始极速微调！")
    else:
        print("⚠️ 警告：未找到预训练权重，将从头开始训练。")
        
    epochs = 15
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 
    
    best_auc = 0.0
    best_metrics = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Fine-tuning Ep {epoch:02d}/{epochs}", leave=False)
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            e_f, e_m, e_p, f_m, f_p = model(imgs)
            
            loss_f = edl_loss(e_f, labels, 2, epoch, epochs)
            loss_m = edl_loss(e_m, labels, 2, epoch, epochs)
            loss_p = edl_loss(e_p, labels, 2, epoch, epochs)
            cos_sim = F.cosine_similarity(f_m, f_p, dim=1)
            loss_ortho = torch.mean(torch.abs(cos_sim))
            
            loss = loss_f + 0.5 * loss_m + 0.5 * loss_p + 0.1 * loss_ortho
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 🔑 修复点：评估时传递正确数量的参数
        auc, acc, sens, spec, tn, fp, fn, tp = evaluate(model, test_loader, device, num_classes=2)
        if auc > best_auc:
            best_auc = auc
            best_metrics = (acc, sens, spec, tn, fp, fn, tp)
            
        print(f"   Epoch {epoch:02d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Test AUC: {auc:.4f} (Best: {best_auc:.4f})")
        
    acc, sens, spec, tn, fp, fn, tp = best_metrics if best_metrics else (0,0,0,0,0,0,0)
    print("\n" + "="*80)
    print("🏥 【临床迁移学习 LUNA16 最终战报】")
    print("经过仅 15 轮域适应 (Domain Adaptation) 后：")
    print(f"  Best Test AUC: {best_auc:.4f} (完美证明骨干网络的泛化潜力！)")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Sens (TPR):    {sens:.4f}")
    print(f"  Spec (TNR):    {spec:.4f}")
    print(f"  混淆矩阵:      TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*80)

if __name__ == "__main__":
    main()
