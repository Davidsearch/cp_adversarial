import torch
import torch.nn as nn
from torchvision import models, transforms
from CleanData import organize_val_data, tiny_imagenet_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime


# --- 全局超参数（与攻击器参数一致）---
epsilon = 3.0       # 降低总扰动预算，迫使算法更高效利用扰动
block_size = 32     # 增大块尺寸以提升稀疏性
num_samples = 2     # 减少每轮采样块数
max_iter = 200      # 增加迭代次数以补偿采样数减少
mu = 2              # 减少成功样本数
num_test_samples = 10
target_class = None
lambda_sparsity = 0.3  # 稀疏性正则化权重

class SZOAttack:
    def __init__(self, model, image, label, block_size=32, target_class=None):
        self.model = model
        self.original_image = image.clone()
        self.label = label
        self.block_size = block_size
        self.image_size = image.shape[-1]
        self.num_blocks_h = self.image_size // block_size
        self.num_blocks_w = self.image_size // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w
        self.prob = torch.ones(self.total_blocks) / self.total_blocks
        self.best_adv = None
        self.best_loss = -np.inf
        self.normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975], 
            std=[0.2770, 0.2691, 0.2821]
        )
        self.target_class = target_class

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
            # 根据块概率调整扰动强度
            delta = torch.randn_like(self.original_image) * (self.prob[b]**0.5 * 0.2)
            delta = self.block_projection(delta, b)
            deltas.append(delta)
        
        combined_delta = torch.stack(deltas).sum(dim=0)
        combined_delta = combined_delta * min(1.0, epsilon / combined_delta.abs().sum())
        return combined_delta.unsqueeze(0), blocks

    def evaluate(self, adv_images):
        with torch.no_grad():
            logits = self.model(adv_images)
            batch_size = logits.size(0)
            
            # 原始损失计算
            if self.target_class is None:
                target = torch.full((batch_size,), self.label, device=logits.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)
            else:
                target = torch.full((batch_size,), self.target_class, device=logits.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target)
            
            # 稀疏性正则化项
            sparsity_penalty = torch.norm(adv_images - self.original_image, p=1, dim=(1,2,3))
            losses += lambda_sparsity * sparsity_penalty
            
        return losses

    def update_prob(self, blocks, losses):
        if self.target_class is None:
            success_indices = losses.argsort()[:mu]
        else:
            success_indices = losses.argsort(descending=True)[:mu]
        success_blocks = blocks[success_indices]
        
        # 强化聚焦策略
        self.prob = torch.zeros_like(self.prob)
        unique_blocks, counts = torch.unique(success_blocks, return_counts=True)
        for b, cnt in zip(unique_blocks, counts):
            self.prob[b] = cnt.float()**1.5  # 非线性加权
        
        # 更小的探索因子
        self.prob = self.prob + 0.005 * torch.ones_like(self.prob)
        self.prob = self.prob / self.prob.sum()

    def attack(self):
        original_image_norm = self.normalize(self.original_image.unsqueeze(0))
        with torch.no_grad():
            original_prob = torch.softmax(self.model(original_image_norm), dim=1)[0]
            print(f"Original prob (true class {self.label}): {original_prob[self.label]:.4f}")

        # 使用简化的进度条格式（直接显示postfix字符串）
        progress_bar = tqdm(
            range(max_iter), 
            desc="Attacking", 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]",  # 简化格式
            # postfix="L0: N/A, L1: N/A"  # 初始化为字符串
        )
        try:
            for t in progress_bar:
                self.current_iter = t
                delta, blocks = self.sample_perturbations()
                adv_images = torch.clamp(self.original_image + delta, 0, 1)
                adv_images_norm = self.normalize(adv_images)
                
                losses = self.evaluate(adv_images_norm)
                best_idx = 0
                
                current_loss = losses[best_idx].item()
                if current_loss > self.best_loss:
                    self.best_loss = current_loss
                    self.best_adv = adv_images[best_idx]
                
                self.update_prob(blocks, losses)
    
                # # 将关键指标写入日志文件
                # logging.info(
                #     f"Iter {t+1} | "
                #     f"Best Loss: {self.best_loss:.4f} | "
                #     f"True Prob: {true_prob:.4f} | "
                #     f"Other Prob: {other_prob:.4f} | "
                #     f"L0: {l0_norm} | L1: {l1_norm:.1f}"
                # )
                
                # 实时稀疏性监控
                with torch.no_grad():
                    delta_map = (adv_images - self.original_image).abs().sum(dim=0)
                    l0_norm = (delta_map > 0.01).sum().item()  # 非零像素数
                    l1_norm = delta_map.sum().item()
                    
    
            return self.best_adv
        except:
            print("\n捕获到中断信号，正在清理显存...")
            torch.cuda.empty_cache()
            sys.exit(0)
            

def main():
    # --- 模型加载与数据预处理 ---
    data_root = "./tiny-imagenet-200"
    organize_val_data(data_root)
    train_loader, test_loader, _ = tiny_imagenet_loader(batch_size=1, data_dir=data_root)
    
    # 加载并适配模型
    model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 200)
    model.features = nn.Sequential(
        *list(model.features.children())[:-1],
        nn.AdaptiveAvgPool2d((7, 7))
    )
    model.eval()
    
    # --- 执行攻击实验 ---
    attack_success = 0
    # test_progress = tqdm(
    #     enumerate(test_loader), 
    #     total=min(num_test_samples, len(test_loader)),
    #     desc="Testing Samples",
    #     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Success: {postfix}]",
    #     postfix="0.0%"
    # )
        # 修改进度条初始化（使用字典格式postfix）
    # 修改后（正确）：
    test_progress = tqdm(
        enumerate(test_loader), 
        total=min(num_test_samples, len(test_loader)),
        desc="Testing Samples",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Success: {postfix}]",  # 直接使用{postfix}
        postfix="0.0%"  # 初始化为字符串
    )
    # 新增稀疏性统计变量
    total_l0 = 0  # 累积所有样本的L0范数
    total_l1 = 0.0  # 累积所有样本的L1范数
    for i, (images, labels) in test_progress:
        if i >= num_test_samples: break
        
        attacker = SZOAttack(
            model=model,
            image=images.squeeze(0),
            label=labels.item(),
            block_size=block_size,
            target_class=target_class
        )
        adv_image = attacker.attack()
        
        with torch.no_grad():
            pred = model(attacker.normalize(adv_image.unsqueeze(0))).argmax().item()
            attack_success += int(pred != labels.item())
        # 计算当前样本的稀疏性指标
        delta_map = (adv_image - images.squeeze(0)).abs().sum(0)
        l0 = (delta_map > 0.01).sum().item()  # 非零像素数
        l1 = delta_map.sum().item()           # 总扰动强度
        total_l0 += l0
        total_l1 += l1
                
        # 可视化首个样本（修复colorbar错误）
        # 修改后的可视化部分
        if i == 0:
            inv_norm = transforms.Normalize(
                mean=[-0.4802/0.2770, -0.4481/0.2691, -0.3975/0.2821],
                std=[1/0.2770, 1/0.2691, 1/0.2821]
            )
            
            # 处理原始图像
            original_inv = inv_norm(images.squeeze(0))
            original_inv = torch.clamp(original_inv, 0, 1)  # 原始图像也裁剪
            
            # 处理对抗样本
            adv_inv = inv_norm(adv_image)
            adv_inv = torch.clamp(adv_inv, 0, 1)  # 关键修复：限制动态范围
            
            delta = (adv_image - images.squeeze(0)).abs().sum(0).numpy()
            
            plt.figure(figsize=(15,5))
            
            # 原始图像
            plt.subplot(131)
            plt.imshow(original_inv.permute(1,2,0).detach().numpy())  # 确保使用裁剪后的数据
            plt.title(f"Original (True: {labels.item()})")
            
            # 对抗样本
            from skimage import exposure
            import cv2
            
            # 转换为OpenCV所需的uint8格式
            adv_np = (adv_inv.permute(1,2,0).numpy() * 255).astype(np.uint8)  # 范围[0,255]
            
            # 转换到YCrCb颜色空间
            ycrcb = cv2.cvtColor(adv_np, cv2.COLOR_RGB2YCrCb)
            
            # 提取Y通道并处理
            y_channel = ycrcb[:,:,0].copy()  # 必须创建副本
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            y_clahe = clahe.apply(y_channel)
            
            # 合并通道
            ycrcb[:,:,0] = y_clahe
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            
            # 转回浮点格式供Matplotlib显示
            result = result.astype(np.float32) / 255.0
            
            # 显示结果
            plt.subplot(132)
            plt.imshow(result)
            plt.title(f"Adversarial new (Pred: {pred})")
            
            # 热力图
            plt.subplot(133)
            im = plt.imshow(delta, cmap='hot', vmax=epsilon*0.1)
            cbar = plt.colorbar(im)
            cbar.set_label("Perturbation Intensity")
            plt.title(f"Sparse Perturbation (L0: {(delta>0.01).sum()})")
            
            plt.tight_layout()
            plt.savefig("sparse_attack_demo.png", dpi=150, bbox_inches='tight')
            plt.close()
        # 修改为字典格式更新
        current_sr = attack_success / (i + 1) * 100
        # 修改后（正确）：
        test_progress.set_postfix(Success=f"{current_sr:.1f}%")  # 使用关键字参数

    # 打印最终统计结果
    avg_l0 = total_l0 / num_test_samples
    avg_l1 = total_l1 / num_test_samples
    print(f"\n最终攻击成功率: {attack_success/num_test_samples*100:.1f}%")
    print(f"平均扰动稀疏性 | L0: {total_l0/num_test_samples:.1f}, L1: {total_l1/num_test_samples:.1f}")

if __name__ == "__main__":
    main()