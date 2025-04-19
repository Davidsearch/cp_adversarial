import torch
import torch.nn as nn
from torchvision import models, transforms
from CleanData import organize_val_data, tiny_imagenet_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SZOAttack:
    def __init__(self, model, image, label, block_size=8):
        self.model = model
        self.original_image = image.clone()  # [C, H, W]
        self.label = label
        self.block_size = block_size
        self.image_size = image.shape[-1]
        # 动态计算分块数
        self.num_blocks_h = self.image_size // block_size
        self.num_blocks_w = self.image_size // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w
        # 初始化块概率
        self.prob = torch.ones(self.total_blocks)
        self.prob /= self.prob.sum()
        self.best_adv = None
        self.best_loss = -np.inf
        self.normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975], 
            std=[0.2770, 0.2691, 0.2821]
        )

    def block_projection(self, delta, block_idx):
        """ 将扰动投影到指定块，其余区域置零 """
        h = block_idx // self.num_blocks_w
        w = block_idx % self.num_blocks_w
        delta_block = torch.zeros_like(delta)
        h_start = h * self.block_size
        w_start = w * self.block_size
        # 打印调试信息
        # print(f"Block {block_idx}: h={h}, w={w}, h_start={h_start}, w_start={w_start}")
        delta_block[:, h_start:h_start+self.block_size, w_start:w_start+self.block_size] = \
            delta[:, h_start:h_start+self.block_size, w_start:w_start+self.block_size]
        return delta_block

    def clip_perturbation(self, delta):
        """ 约束扰动L1范数不超过epsilon """
        l1_norm = torch.sum(torch.abs(delta))
        if l1_norm > epsilon:
            delta = delta * epsilon / l1_norm
        return delta


    def sample_perturbations(self):
        blocks = torch.multinomial(self.prob, num_samples, replacement=True)
        deltas = []
        for b in blocks:
            delta = torch.randn_like(self.original_image)
            delta = self.block_projection(delta, b)
            delta = delta / torch.sum(torch.abs(delta)) * epsilon  # 强制L1范数为epsilon
            deltas.append(delta)
        return torch.stack(deltas), blocks

    def evaluate(self, adv_images):
        with torch.no_grad():
            logits = self.model(adv_images)
            batch_size = logits.size(0)
            
            if target_class is None:
                # 非定向攻击：最小化真实类别的概率
                target = torch.full((batch_size,), self.label, device=logits.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)  # 保留每个样本的独立损失
            else:
                # 定向攻击：最大化目标类别的概率
                target = torch.full((batch_size,), target_class, device=logits.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target)  # 取负号以最大化目标概率
            return losses

    def update_prob(self, blocks, losses):
        if target_class is None:
            success_indices = losses.argsort()[:mu]
        else:
            success_indices = losses.argsort(descending=True)[:mu]
        success_blocks = blocks[success_indices]
        
        unique_blocks, counts = torch.unique(success_blocks, return_counts=True)
        for b, cnt in zip(unique_blocks, counts):
            self.prob[b] += cnt.item()
        
        self.prob = self.prob + 0.1 * torch.ones_like(self.prob)  # 探索因子
        self.prob = self.prob / self.prob.sum()

    def attack(self):
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # original_image_norm = self.normalize(self.original_image.unsqueeze(0))
        '''
        上面两行是先前的代码逻辑，可能存在问题：
        数据加载阶段：Tiny-ImageNet 使用专用的归一化参数（mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821]）。

        攻击阶段：SZOAttack 内部使用了 ImageNet 的归一化参数（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）。
        '''
        original_image_norm = self.normalize(self.original_image.unsqueeze(0))
        with torch.no_grad():
            original_prob = torch.softmax(self.model(original_image_norm), dim=1)[0]
            print(f"Original prob (true class {self.label}): {original_prob[self.label]:.4f}")

        # 创建进度条（添加更多信息槽位）
        progress_bar = tqdm(
        range(max_iter), 
        desc="Attacking", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}<{remaining}]",
        postfix={"Best Loss": "N/A", "True Prob": "N/A", "Other Prob": "N/A"}
    )

        for t in progress_bar:
            deltas, blocks = self.sample_perturbations()
            adv_images = torch.clamp(self.original_image + deltas, 0, 1)
            adv_images_norm = self.normalize(adv_images)
            
            losses = self.evaluate(adv_images_norm)
            best_idx = losses.argmin() if target_class is None else losses.argmax()
            current_loss = losses[best_idx].item()
            
            if current_loss > self.best_loss:
                self.best_loss = current_loss
                self.best_adv = adv_images[best_idx]
            
            self.update_prob(blocks, losses)
    
    
            # --- 动态更新进度条中的调试信息 ---
            with torch.no_grad():
                adv_probs = torch.softmax(self.model(adv_images_norm[best_idx].unsqueeze(0)), dim=1)
                true_prob = adv_probs[0, self.label].item()
                other_prob = adv_probs[0, torch.arange(adv_probs.shape[1]) != self.label].max().item()
            
            # 更新进度条右侧信息
            progress_bar.set_postfix({
                "Best Loss": f"{self.best_loss:.4f}",
                "True Prob": f"{true_prob:.4f}",
                "Other Prob": f"{other_prob:.4f}"
            }, refresh=False)
    
        return self.best_adv

import torch
from torchvision import models, transforms
from CleanData import organize_val_data, tiny_imagenet_loader
import matplotlib.pyplot as plt
import numpy as np

# 超参数配置
epsilon = 10.0              # L1扰动约束
block_size = 8              # 分块大小
num_samples = 50            # 每轮采样数
max_iter = 50               # 最大迭代次数
mu = 10                     # 成功样本数（用于更新均值）
num_test_samples = 10       # 测试样本数量
target_class = None         # None为非定向攻击，指定类别则为定向攻击

def main():
    # --- Step 1: 整理数据集并加载 ---
    data_root = "./tiny-imagenet-200"  # 数据集根目录
    batch_size = 1                     # 单样本攻击
    
    # 整理验证集文件结构
    organize_val_data(data_root)
    
    # 加载Tiny-ImageNet数据集
    train_loader, test_loader, num_classes = tiny_imagenet_loader(
        batch_size=batch_size,
        data_dir=data_root
    )
    
    # --- Step 2: 加载模型并适配输入 ---
    # 加载VGG11模型（预训练模型需支持动态输入或调整输入层）
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 200)  # 适配 Tiny-ImageNet 的 200 个类别
    model.features = nn.Sequential(
    *list(model.features.children())[:-1],  # 移除最后一个 MaxPool 层
    nn.AdaptiveAvgPool2d((7, 7))           # 适配 64x64 输入
)
    model.eval()
    
    # --- Step 3: 定义反标准化函数（用于可视化） ---
    inv_normalize = transforms.Normalize(
        mean=[-0.4802/0.2770, -0.4481/0.2691, -0.3975/0.2821],
        std=[1/0.2770, 1/0.2691, 1/0.2821]
    )
    
    # --- Step 4: 执行批量对抗攻击实验 ---
    attack_success = 0
    total_samples = min(num_test_samples, len(test_loader))

        # 添加外层进度条（遍历测试样本）
    test_progress = tqdm(
        enumerate(test_loader), 
        total=total_samples, 
        desc="Testing Samples"
    )
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_test_samples:
            break
        
        # # 调整输入尺寸至224x224（兼容VGG11）
        # resize_transform = transforms.Compose([
        #     transforms.Resize(224),    # 上采样至224x224
        #     transforms.CenterCrop(224)
        # ])
        # image_pil = transforms.ToPILImage()(images.squeeze(0))
        # image_resized = resize_transform(image_pil)
        # image_tensor = transforms.ToTensor()(image_resized)
            
        # 直接使用64x64输入（无需上采样）
        image_tensor = images.squeeze(0)
        
        # 初始化攻击器
        attacker = SZOAttack(
            model=model,
            image=image_tensor,
            label=labels.item(),
            block_size=block_size
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
        
        # --- Step 5: 可视化结果（示例输出第一张） ---
        if i == 0:
            plt.figure(figsize=(12, 4))
            
            # 原始图像（反标准化至Tiny-ImageNet参数）
            plt.subplot(1, 3, 1)
            original_vis = inv_normalize(image_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
            plt.imshow(original_vis)
            plt.title(f"Original\nTrue: {labels.item()}")
            
            # 对抗样本
            plt.subplot(1, 3, 2)
            adv_vis = inv_normalize(adv_image).clamp(0, 1).permute(1, 2, 0).numpy()
            plt.imshow(adv_vis)
            plt.title(f"Adversarial\nPred: {pred_class}")
            
            # 扰动热力图
            plt.subplot(1, 3, 3)
            delta = (adv_image - image_tensor).abs().sum(dim=0).numpy()
            plt.imshow(delta, cmap="hot")
            plt.colorbar()
            plt.title(f"Perturbation (L1={delta.sum():.1f})")
            
            plt.tight_layout()
            plt.savefig("attack_demo.png")
            plt.show()
    
        # 更新外层进度条（显示当前攻击成功率）
        current_success_rate = attack_success / (i + 1) * 100
        test_progress.set_postfix({
            "Success Rate": f"{current_success_rate:.1f}%"
        })

if __name__ == "__main__":
    main()