import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms, datasets
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math

# GMM-EM特需1
import sklearn
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

# --- 全局超参数 ---
epsilon = 0.3       # 总扰动预算
block_size = 16     # 块尺寸
num_samples = 5     # 每轮采样块数
max_iter = 15       # 增加迭代次数以观察趋势
mu = 3              # 成功样本保留数
num_test_classes = 1  # 测试类别数
target_class = None  
lambda_sparsity = 0.3  # 稀疏性正则化系数

# --- CMA-ES 参数 ---
cma_lambda = 10     # 种群大小
cma_mu = 5          # 父代数量
cma_sigma = 0.5     # 初始步长

img_size = 64
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
])

val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
])

NORMALIZE_MEAN = [0.4802, 0.4481, 0.3975]
NORMALIZE_STD = [0.2770, 0.2691, 0.2821]

def normalize(x, device):
    """全局归一化函数"""
    mean = torch.tensor(NORMALIZE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------- 数据加载 ---------------------------
def get_tinyimagenet_loader(data_dir, batch_size=64, train=False):
    """加载Tiny ImageNet数据集"""
    transform = val_transform if not train else train_transform
    dataset_path = os.path.join(data_dir, 'train' if train else 'val')
    
    checkpoint_path = os.path.join(dataset_path, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

class GMMProbabilityUpdater:
    def __init__(self, num_blocks, num_blocks_w, n_components=3, max_features=5):
        self.n_components = n_components
        self.max_features = max_features
        self.features = {
            'block_id': [],
            'position_x': [],
            'position_y': [],
            'success_count': [],
            'recent_selected': [],
            'loss_change': []
        }
        self.gmm = GaussianMixture(n_components=self.n_components, max_iter=100)
        self.num_blocks_w = num_blocks_w
        self.num_blocks_h = num_blocks // num_blocks_w
        self.block_probs = np.ones(num_blocks) / num_blocks
    
    def update_features(self, blocks, losses):
        """收集块特征"""
        block_centers = [(bid // self.num_blocks_w, bid % self.num_blocks_w)
                        for bid in blocks.cpu().numpy()]

        for bid, (cx, cy) in zip(blocks, block_centers):
            self.features['block_id'].append(bid)
            self.features['position_x'].append(cx)
            self.features['position_y'].append(cy)
            self.features['success_count'].append(1)
            self.features['recent_selected'].append(1)
            self.features['loss_change'].append(losses.mean().item())

        print("特征统计：")
        print(f"Min: {self.features.min(axis=0)}")
        print(f"Max: {self.features.max(axis=0)}")
        print(f"NaN数量：{np.isnan(self.features).sum()}")
    
    def train_gmm(self):
        if len(self.features['block_id']) < self.max_features:
            print(f"训练跳过：当前特征数({len(self.features['block_id'])}) < 最小要求({self.max_features})")
            return

        features = np.column_stack([
            np.array(self.features['position_x']),
            np.array(self.features['position_y']),
            np.array(self.features['success_count']),
            np.array(self.features['recent_selected']),
            np.array(self.features['loss_change'])
        ])
        
        # 增强型数据清洗
        if features.size == 0:
            print("⚠️ 特征矩阵为空，跳过训练")
            return
            
        nan_mask = np.isnan(features).any(axis=1)
        print(f"清理前：总样本数={len(features)}, NaN样本数={np.sum(nan_mask)}")
        
        features = features[~nan_mask]
        print(f"清理后有效样本数={len(features)}")
        
        if len(features) < 2:  # GaussianMixture至少需要2个样本
            print("⚠️ 有效样本不足，跳过训练")
            return
        
        try:
            self.gmm = GaussianMixture(n_components=self.n_components, max_iter=100)
            self.gmm.fit(features)
            print("✅ GMM训练成功")
        except Exception as e:
            print(f"GMM训练失败: {str(e)}")
    
    def calculate_probs(self, num_blocks):
        """计算块概率"""
        all_blocks = np.arange(num_blocks)
        block_centers = [(bid // self.num_blocks_w, bid % self.num_blocks_w)
                        for bid in all_blocks]

        X = np.zeros((num_blocks, 5))
        for bid in all_blocks:
            cx, cy = block_centers[bid]
            X[bid] = [
                cx,
                cy,
                self.features['success_count'].count(bid),
                np.mean(self.features['recent_selected'][-10:]),
                np.mean([x for x, b in zip(self.features['loss_change'], 
                                         self.features['block_id']) if b == bid])
            ]
        
        probs = self.gmm.predict_proba(X)
        return probs.mean(axis=1)

class DebugVGG16(nn.Module):
    """加载本地预训练的VGG16模型"""
    def __init__(self, num_classes=200):
        super().__init__()
        original_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = original_model.features

        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.features(dummy)
            in_features = features.view(-1).shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        pretrained_dict = torch.load("./train_vgg16/best_vgg16.pth", map_location="cpu")
        self.load_state_dict(pretrained_dict)
        missing, unexpected = self.load_state_dict(pretrained_dict, strict=False)
        print("缺失的键（未加载的权重）:", missing)
        print("意外的键（多余的权重）:", unexpected)
        print("成功加载本地VGG16权重")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CMAESBlock:
    """管理每个块的CMA-ES优化器"""
    def __init__(self, block_size):
        self.dim = block_size
        self.lambda_ = int(4 + 3 * math.log(block_size))
        self.mu = max(1, self.lambda_ // 2)
        self.mean = np.zeros(block_size)
        self.C = np.eye(block_size)
        self.sigma = cma_sigma
        self.p_sigma = np.zeros(block_size)
        self.p_c = np.zeros(block_size)
        self.mu = cma_mu
        self.lambda_ = cma_lambda
        self.cc = 4 / (self.dim + 4)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.cmu = 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu)
        self.damps = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1))-1) + self.cc
        self.chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i+1) for i in range(self.mu)])
        self.weights = self.weights / self.weights.sum()

    def sample(self):
        """生成候选样本"""
        return [self.sigma * np.random.multivariate_normal(self.mean, self.C) 
                for _ in range(self.lambda_)]

    def update(self, parents):
        """更新参数"""
        if len(parents) != self.mu:
            raise ValueError("Parents数量必须等于mu")

        weighted_mean = np.zeros_like(self.mean)
        for w, x in zip(self.weights, parents):
            weighted_mean += w * x
        new_mean = weighted_mean

        y = (new_mean - self.mean) / self.sigma
        self.p_sigma = (1 - self.cc) * self.p_sigma + np.sqrt(self.cc*(2-self.cc)*self.mu) * y
        sigma_norm = np.linalg.norm(self.p_sigma) / self.chi_n
        self.sigma *= np.exp((sigma_norm - 1) * self.cc / self.damps)
        self.p_c = (1 - self.cc) * self.p_c + np.sqrt(self.cc*(2-self.cc)*self.mu) * y
        delta_C = np.outer(self.p_c, self.p_c) * self.c1
        for x in parents:
            z = (x - self.mean) / self.sigma
            delta_C += self.cmu * np.outer(z, z)
        self.C = (1 - self.c1 - self.cmu) * self.C + delta_C
        self.mean = new_mean.copy()
        self.C = (self.C + self.C.T) / 2

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at score {score} after {self.counter} iterations without improvement.")
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

class SZOAttack:
    """稀疏区域优化的对抗攻击"""
    def __init__(self, model, image, label, block_size, device, epsilon=0.1, universal_perturbation=None):
        self.device = image.device
        self.model = model.to(self.device)
        self.original_image = image.clone().detach().to(self.device)
        self.label = label.to(self.device)
        self.block_size = block_size
        self.epsilon = epsilon
        self.H, self.W = image.shape[-2:]
        self.num_blocks_h = (self.H + block_size - 1) // block_size
        self.num_blocks_w = (self.W + block_size - 1) // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w

        try:
            self.gmm_updater = GMMProbabilityUpdater(
                num_blocks=self.total_blocks,
                num_blocks_w=self.num_blocks_w,
                n_components=3,
                max_features=5
            )
            print("✅ GMM初始化成功")
        except Exception as e:
            raise RuntimeError(f"GMM初始化失败: {str(e)}")

        print(self.gmm_updater.gmm)
        self.prob = torch.ones(self.total_blocks, device=self.device) / self.total_blocks
        self.block_history = []
        self.loss_history = []
        self.best_adv = torch.empty_like(self.original_image, device=self.device)
        self.best_loss = -np.inf
        self.target_class = None
        self.global_acc_history = []
        self.eval_interval = 1
        self.cma_blocks = {bid: CMAESBlock(block_size) for bid in range(self.total_blocks)}

        if universal_perturbation is None:
            self.universal_perturbation = torch.zeros_like(image.unsqueeze(0), device=device)
        else:
            self.universal_perturbation = universal_perturbation.clone()
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    def block_projection(self, delta, bid):
        """全GPU化的块投影"""
        num_blocks_per_row = (self.W + self.block_size - 1) // self.block_size
        h_start = (bid // num_blocks_per_row) * self.block_size
        w_start = (bid % num_blocks_per_row) * self.block_size
        
        h_slice = slice(max(0, h_start), min(h_start+self.block_size, self.H))
        w_slice = slice(max(0, w_start), min(w_start+self.block_size, self.W))
        perturbation = torch.randn(3, h_slice.stop-h_slice.start, w_slice.stop-w_slice.start,
                                  device=self.device) * cma_sigma
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        delta[:, h_slice, w_slice].add_(perturbation)
        return delta

    def sample_perturbations(self):
        """批量生成GPU扰动"""
        selected_blocks = torch.multinomial(self.prob, num_samples, replacement=True)
        valid_bids = selected_blocks.unique().cpu().numpy()
        delta = torch.zeros_like(self.original_image, device=self.device)
        
        for bid in valid_bids:
            if bid >= self.total_blocks:
                continue
            delta = self.block_projection(delta, bid)
        
        return delta.to(self.device), selected_blocks
    
    def evaluate(self, adv_images):
        """评估对抗样本"""
        with torch.no_grad():
            logits = self.model(adv_images)
            if self.target_class is None:
                target = torch.full((logits.size(0),), self.label, device=self.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)
            else:
                target = torch.full((logits.size(0),), self.target_class, device=self.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target)
            sparsity_penalty = torch.norm(adv_images - self.original_image, p=1)
            losses += lambda_sparsity * sparsity_penalty
            return losses

    def update_prob(self, blocks, losses):
        """GMM-EM概率更新"""
        self.block_history.extend(blocks.cpu().numpy())
        self.loss_history.append(losses.mean().item())
        
        if len(self.block_history) % 5 == 0:
            features = []
            for bid in np.unique(self.block_history):
                cx = (bid // self.num_blocks_w) / self.num_blocks_h
                cy = (bid % self.num_blocks_w) / self.num_blocks_w
                
                # 安全计算成功率
                success_count = sum(1 for b in self.block_history[-100:] if b == bid)
                success_rate = success_count / 100 if success_count > 0 else 0.0
                
                # 安全计算最近频率
                recent_count = sum(1 for b in self.block_history[-10:] if b == bid)
                recent_freq = recent_count / 10 if recent_count > 0 else 0.0
                
                # 安全计算损失变化
                loss_values = [l for b, l in zip(self.block_history, self.loss_history) if b == bid]
                loss_change = np.nanmean(loss_values) if loss_values else 0.0
                
                features.append([cx, cy, success_rate, recent_freq, loss_change])
            
            # 空数据检查
            if not features:
                print("⚠️ 无有效特征数据，跳过GMM训练")
                return
            
            try:
                features_array = np.array(features)
                # print(f"\n特征矩阵形状: {features_array.shape}")
                
                # 数据清洗
                valid_rows = ~np.isnan(features_array).any(axis=1)
                features_clean = features_array[valid_rows]
                # print(f"\n清洗后特征数: {len(features_clean)}")
                
                if len(features_clean) < 2:
                    print("⚠️ 清洗后样本不足，跳过训练")
                    return
                    
                self.gmm = GaussianMixture(n_components=3)
                self.gmm.fit(features_clean)
                
                # 计算所有块概率
                all_features = []
                for bid in range(self.total_blocks):
                    cx = (bid // self.num_blocks_w) / self.num_blocks_h
                    cy = (bid % self.num_blocks_w) / self.num_blocks_w
                    
                    success = np.nanmean([1 if b == bid else 0 for b in self.block_history]) or 0.0
                    recent = np.nanmean([1 if b == bid else 0 for b in self.block_history[-10:]]) or 0.0
                    loss_values = [l for b, l in zip(self.block_history, self.loss_history) if b == bid]
                    loss = np.nanmean(loss_values) if loss_values else 0.0
                    
                    all_features.append([cx, cy, success, recent, loss])
                
                all_features = np.array(all_features)
                all_features = np.nan_to_num(all_features, nan=0.0)  # 最终安全处理
                
                probs = self.gmm.predict_proba(all_features)
                self.prob = torch.tensor(probs.mean(axis=1), device=self.device, dtype=torch.float32)
                self.prob = 0.9 * self.prob + 0.1 * torch.ones_like(self.prob)/self.total_blocks
                self.prob /= self.prob.sum()
                
            except Exception as e:
                print(f"GMM更新失败: {str(e)}")
                import traceback
                traceback.print_exc()


    def attack(self, val_loader, device, enable_eval=True, show_progress=True):
        """全GPU化攻击流程"""
        progress_bar = tqdm(range(max_iter), desc="🔥GPU攻击进程") if show_progress else range(max_iter)
        
        for t in progress_bar:
            delta, blocks = self.sample_perturbations()
            delta = delta.unsqueeze(0)
            self.universal_perturbation = torch.clamp(
                self.universal_perturbation + delta,
                -self.epsilon,
                self.epsilon
            )

            adv_images = self.original_image + self.universal_perturbation
            losses = self.evaluate(adv_images)
            self.update_prob(blocks, losses)

            with torch.no_grad():
                outputs = self.model(adv_images)
                loss = nn.CrossEntropyLoss()(outputs, self.label.view(-1))
                loss += lambda_sparsity * torch.norm(delta, p=1)
            
            if loss > self.best_loss:
                self.best_loss = loss
                self.best_adv = adv_images
            
            if enable_eval and (t % self.eval_interval == 0):
                attacked_val_images = []
                attacked_val_labels = []
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    adv_images = self.attack_batch(images, labels)
                    attacked_val_images.append(adv_images)
                    attacked_val_labels.append(labels)

                attacked_val_images = torch.cat(attacked_val_images, dim=0)
                attacked_val_labels = torch.cat(attacked_val_labels, dim=0)
                attacked_dataset = TensorDataset(attacked_val_images, attacked_val_labels)
                attacked_dataloader = DataLoader(attacked_dataset, batch_size=64, shuffle=False)
                current_acc = evaluate_global_accuracy(self.model, attacked_dataloader, device)
                self.global_acc_history.append(current_acc)
                
                if current_acc > 0 and self.early_stopping(current_acc):
                    print(f"Early stopping at iteration {t} due to no improvement in global accuracy.")
                    break

            global_acc = np.mean(self.global_acc_history)
            if enable_eval:
                progress_bar.set_postfix(
                    Global_Acc=f"{global_acc:.2f}%" if enable_eval else "N/A", 
                    Current_Acc=f"{current_acc:.2f}%", 
                    Loss=f"{loss.item():.4f}"
                )
        
        if len(self.global_acc_history) < max_iter:
            self.global_acc_history.extend([self.global_acc_history[-1]] * (max_iter - len(self.global_acc_history)))
        
        return self.best_adv.squeeze(0)

    def attack_batch(self, images, labels):
        """全GPU批量攻击"""
        images = images.to(self.device)
        labels = labels.to(self.device)
        adv_images = torch.empty_like(images, device=self.device)
        
        for i in range(images.shape[0]):
            perturbed_img = images[i] + self.universal_perturbation
            adv_images[i] = perturbed_img.squeeze(0)
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
        return adv_images

def evaluate_global_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.cuda.amp.autocast(): 
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./tiny-imagenet-200"
    val_loader = get_tinyimagenet_loader(data_root, train=False)
    
    model = DebugVGG16(num_classes=200).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print("输出张量形状:", output.shape)
        features = model.features(dummy_input)
        print(f"特征图尺寸: {features.shape}")
        flattened = torch.flatten(features, 1)
        print(f"展平后维度: {flattened.shape}")
    
    orig_acc = evaluate_global_accuracy(model, val_loader, device)
    print(f"\n原始模型准确率: {orig_acc:.2f}%")
    
    class_samples = {}
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        for img, lbl in zip(images, labels):
            lbl_value = lbl.item()
            if lbl_value not in class_samples:
                class_samples[lbl_value] = (img, lbl)
        if len(class_samples) >= num_test_classes:
            break
    
    attack_success = 0
    total_l0, total_l1 = 0, 0.0
    universal_perturbation = torch.zeros((1,3,64,64), device=device)
    all_class_history = {}

    test_progress = tqdm(class_samples.items(), desc="攻击进度")
    
    for class_id, (image, label) in test_progress:
        attacker = SZOAttack(
            model=model,
            image=image,
            label=label,
            block_size=block_size,
            device=device,
            epsilon=epsilon,
            universal_perturbation=universal_perturbation
        )
        adv_img = attacker.attack(val_loader, device)
        all_class_history[class_id] = {
            'global_acc': attacker.global_acc_history,
            'iterations': list(range(0, max_iter, attacker.eval_interval))
        }

        for class_id, data in all_class_history.items():
            plt.figure(figsize=(10, 5))
            plt.plot(
                data['iterations'], 
                data['global_acc'], 
                marker='o', 
                label=f'Class {class_id} Attack'
            )
            plt.axhline(y=orig_acc, color='r', linestyle='--', label='Original Accuracy')
            plt.title(f"Global Accuracy During Class {class_id} Attack")
            plt.xlabel("Attack Iteration")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"global_acc_class_{class_id}.png")
            plt.close()

        universal_perturbation = attacker.universal_perturbation
        
        with torch.no_grad():
            logits = model(adv_img.unsqueeze(0))
            pred = logits.argmax().item()
            attack_success += int(pred != label.item())
        
        delta = (adv_img - image).abs().sum(0)
        total_l0 += (delta > 0.005).sum().item()
        total_l1 += delta.sum().item()
        
        current_sr = attack_success / (list(class_samples.keys()).index(class_id) + 1) * 100
        test_progress.set_postfix(Success=f"{current_sr:.1f}%")
    
    print(f"\n最终攻击成功率: {attack_success/len(class_samples)*100:.1f}%")
    print(f"平均L0扰动: {total_l0/len(class_samples):.1f}, 平均L1扰动: {total_l1/len(class_samples):.1f}")

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_class_history)))

    for (class_id, data), color in zip(all_class_history.items(), colors):
        if len(data['global_acc']) < max_iter:
            data['global_acc'].extend([data['global_acc'][-1]] * (max_iter - len(data['global_acc'])))
        if len(data['iterations']) < max_iter:
            data['iterations'].extend([data['iterations'][-1]] * (max_iter - len(data['iterations'])))

        plt.plot(
            data['iterations'], 
            data['global_acc'], 
            marker='o', 
            color=color,
            linewidth=2,
            markersize=8,
            label=f'Class {class_id}'
        )

    plt.axhline(y=orig_acc, color='black', linestyle='--', label='Baseline')
    plt.title("Global Accuracy Under Multi-Class Attacks")
    plt.xlabel("Attack Iteration")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_classes_accuracy_trend.png", bbox_inches='tight')

    with open('attack_history.json', 'w') as f:
        json.dump(all_class_history, f)

if __name__ == "__main__":
    main()
