import torch
import torch.nn as nn
from torchvision import models, transforms
from CleanData import organize_val_data, tiny_imagenet_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 全局超参数（与攻击器参数一致）---
epsilon = 5.0       # 总扰动预算（关键参数）
block_size = 16     # 增大块尺寸以提升稀疏性
num_samples = 5     # 减少每轮采样块数
max_iter = 100      # 增加迭代次数以补偿采样数减少
mu = 3              # 减少成功样本数
num_test_samples = 10
target_class = None

class SZOAttack:
    def __init__(self, model, image, label, block_size=16, target_class=None):  # 增大块尺寸
        self.model = model
        self.original_image = image.clone()
        self.label = label
        self.block_size = block_size
        self.image_size = image.shape[-1]
        self.num_blocks_h = self.image_size // block_size
        self.num_blocks_w = self.image_size // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w
        self.prob = torch.ones(self.total_blocks) / self.total_blocks  # 初始均匀分布
        self.best_adv = None
        self.best_loss = -np.inf
        self.normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975], 
            std=[0.2770, 0.2691, 0.2821]
        )
        self.target_class = target_class  # 存储为实例属性

    def block_projection(self, delta, block_idx):
        h = block_idx // self.num_blocks_w
        w = block_idx % self.num_blocks_w
        delta_block = torch.zeros_like(delta)
        h_start = h * self.block_size
        w_start = w * self.block_size
        delta_block[:, h_start:h_start+self.block_size, w_start:w_start+self.block_size] = \
            delta[:, h_start:h_start+self.block_size, w_start:w_start+self.block_size]
        return delta_block

    def sample_perturbations(self):
        blocks = torch.multinomial(self.prob, num_samples, replacement=True)
        deltas = []
        for b in blocks:
            delta = torch.randn_like(self.original_image) * 0.1  # 减小初始扰动幅度
            delta = self.block_projection(delta, b)
            deltas.append(delta)
        
        # 合并所有扰动并整体约束L1范数
        combined_delta = torch.stack(deltas).sum(dim=0)
        l1_norm = torch.sum(torch.abs(combined_delta))
        if l1_norm > epsilon:
            combined_delta = combined_delta * epsilon / l1_norm
        return combined_delta.unsqueeze(0), blocks  # 返回整体扰动

    def evaluate(self, adv_images):
        with torch.no_grad():
            logits = self.model(adv_images)
            batch_size = logits.size(0)
            if self.target_class is None:  # 通过self访问
                target = torch.full((batch_size,), self.label, device=logits.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)
            else:
                target = torch.full((batch_size,), self.target_class, device=logits.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target)
            return losses

    def update_prob(self, blocks, losses):
        if target_class is None:
            success_indices = losses.argsort()[:mu]
        else:
            success_indices = losses.argsort(descending=True)[:mu]
        success_blocks = blocks[success_indices]
        
        # 降低探索因子至0.01
        unique_blocks, counts = torch.unique(success_blocks, return_counts=True)
        self.prob *= 0.0
        for b, cnt in zip(unique_blocks, counts):
            self.prob[b] += cnt.item()
        self.prob = self.prob + 0.01 * torch.ones_like(self.prob)  # 减小探索因子
        self.prob = self.prob / self.prob.sum()

    def attack(self):
        original_image_norm = self.normalize(self.original_image.unsqueeze(0))
        with torch.no_grad():
            original_prob = torch.softmax(self.model(original_image_norm), dim=1)[0]
            print(f"Original prob (true class {self.label}): {original_prob[self.label]:.4f}")

        progress_bar = tqdm(
            range(max_iter), 
            desc="Attacking", 
            postfix={"Best Loss": "N/A", "True Prob": "N/A", "Other Prob": "N/A"}
        )

        for t in progress_bar:
            delta, blocks = self.sample_perturbations()
            adv_images = torch.clamp(self.original_image + delta, 0, 1)
            adv_images_norm = self.normalize(adv_images)
            
            losses = self.evaluate(adv_images_norm)
            best_idx = 0  # 由于整体扰动，只有一个样本
            
            current_loss = losses[best_idx].item()
            if current_loss > self.best_loss:
                self.best_loss = current_loss
                self.best_adv = adv_images[best_idx]
            
            self.update_prob(blocks, losses)
            
            with torch.no_grad():
                adv_probs = torch.softmax(self.model(adv_images_norm), dim=1)
                true_prob = adv_probs[0, self.label].item()
                other_prob = adv_probs[0, torch.arange(adv_probs.shape[1]) != self.label].max().item()
            
            progress_bar.set_postfix({
                "Best Loss": f"{self.best_loss:.4f}",
                "True Prob": f"{true_prob:.4f}",
                "Other Prob": f"{other_prob:.4f}"
            }, refresh=False)

        return self.best_adv

def main():
    # --- Step 1: 整理数据集并加载 ---
    data_root = "./tiny-imagenet-200"
    batch_size = 1
    
    # 整理验证集文件结构
    organize_val_data(data_root)
    
    # 加载Tiny-ImageNet数据集
    train_loader, test_loader, num_classes = tiny_imagenet_loader(
        batch_size=batch_size,
        data_dir=data_root
    )
    
    # # --- Step 2: 加载模型并适配输入 ---
    # model = models.vgg11(pretrained=True)
    # 使用新的weights API加载预训练模型
    from torchvision.models import VGG11_Weights
    
    model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)  # 替换pretrained=True

    
    # 修改模型结构适配64x64输入
    model.classifier[6] = nn.Linear(4096, 200)
    model.features = nn.Sequential(
        *list(model.features.children())[:-1],  # 移除最后一个MaxPool层
        nn.AdaptiveAvgPool2d((7, 7))            # 适配64x64输入
    )
    model.eval()
    
    # --- Step 3: 定义反标准化函数 ---
    inv_normalize = transforms.Normalize(
        mean=[-0.4802/0.2770, -0.4481/0.2691, -0.3975/0.2821],
        std=[1/0.2770, 1/0.2691, 1/0.2821]
    )
    
    # --- Step 4: 执行批量对抗攻击实验 ---
    attack_success = 0
    total_samples = min(num_test_samples, len(test_loader))
    
    # 修改进度条格式字符串
    test_progress = tqdm(
        enumerate(test_loader), 
        total=total_samples, 
        desc="Testing Samples",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Success Rate: {postfix}]",  # 移除索引访问
        postfix={"Success Rate": "0.0%"}  # 初始化postfix为字典
    )
    
    for i, (images, labels) in test_progress:
        if i >= num_test_samples:
            break
        
        # 直接使用64x64输入（无需上采样）
        image_tensor = images.squeeze(0)
        
        # 初始化攻击器（传入调整后的超参数）
        attacker = SZOAttack(
            model=model,
            image=image_tensor,
            label=labels.item(),
            block_size=block_size,  # 使用全局变量 block_size=16
            target_class=target_class  # 传入全局变量target_class
        )
        
        # 执行攻击
        adv_image = attacker.attack()
        
        # 验证攻击效果
        with torch.no_grad():
            adv_input = attacker.normalize(adv_image.unsqueeze(0))
            adv_logits = model(adv_input)
            pred_class = adv_logits.argmax().item()
            if pred_class != labels.item():
                attack_success += 1
        
        # 可视化第一个样本的结果
        if i == 0:
            plt.figure(figsize=(12, 6))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            original_vis = inv_normalize(image_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
            plt.imshow(original_vis)
            plt.title(f"Original (True: {labels.item()})")
            
            # 对抗样本
            plt.subplot(1, 3, 2)
            adv_vis = inv_normalize(adv_image).clamp(0, 1).permute(1, 2, 0).numpy()
            plt.imshow(adv_vis)
            plt.title(f"Adversarial (Pred: {pred_class})")
            
            # 稀疏扰动热力图
            plt.subplot(1, 3, 3)
            delta = (adv_image - image_tensor).abs().sum(dim=0).numpy()
            plt.imshow(delta, cmap="hot", vmax=epsilon*0.2)  # 限制颜色范围以突出稀疏性
            plt.colorbar()
            plt.title(f"Sparse Perturbation\nL1={delta.sum():.1f} (ε={epsilon})")
            
            plt.tight_layout()
            plt.savefig("sparse_attack_demo.png", dpi=200, bbox_inches="tight")
            plt.show()
        
        # 实时更新攻击成功率
        current_success_rate = attack_success / (i + 1) * 100
        test_progress.set_postfix({
            "Success Rate": f"{current_success_rate:.1f}%"
        }, refresh=False)

    print(f"\n最终攻击成功率: {attack_success/num_test_samples*100:.1f}%")
    print(f"扰动稀疏性指标 (L0范数): {delta.sum() > 0.1*epsilon}")  # 检查是否满足稀疏性要求

if __name__ == "__main__":

    
    main()