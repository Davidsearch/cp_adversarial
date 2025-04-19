import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms, datasets
from tqdm import tqdm
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from torch.utils.data import TensorDataset, DataLoader
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
'''
传参思路:

    main[主函数] -->|初始化| universal_perturbation[全局扰动]
    main -->|逐类别循环| attack[攻击实例]
    attack -->|传入| universal_perturbation
    attack -->|更新| universal_perturbation
    evaluate_global_accuracy -->|应用扰动| universal_perturbation
'''
# --- 全局超参数 ---
epsilon = 0.14       # 总扰动预算
block_size = 16     # 块尺寸
num_samples = 5     # 每轮采样块数
max_iter = 50      # 增加迭代次数以观察趋势,这个不应该设的太小(比如3)
mu = 3              # 成功样本保留数
num_test_classes = 20  # 测试类别数（从200类中随机选择）
target_class = None # 当前是无目标攻击 
lambda_sparsity = 0.3  # 稀疏性正则化系数

debug_mode = 1  # 1启用日志和报告，0禁用（设置为0时执行原版逻辑）

# --- CMA-ES 参数 ---
cma_lambda = 10     # 种群大小
cma_mu = 5          # 父代数量
cma_sigma = 0.5     # 初始步长
img_size = 64 #tiny-imagenet的大小是64


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
]) #这里已经进行归一化了，只能有这一次，后面都不许归一化！
# 在 evaluate 和 attack 等函数中不需要再次调用 normalize



# 随机性的限定--使得模型准确率在每次运行的过程中更稳定,便于分析
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保卷积操作确定性
    torch.backends.cudnn.benchmark = False     # 关闭自动优化(避免引入随机性)

# ---------------------------- 数据加载 ---------------------------
def get_tinyimagenet_loader(data_dir, batch_size=64, train=False):
    """加载Tiny ImageNet数据集"""
    transform = val_transform if not train else train_transform  # 根据是否是训练集选择不同的预处理
    # 这里的val_transform是已经进行归一化的
    
    dataset_path = os.path.join(data_dir, 'train' if train else 'val')
    
    # 清理检查点文件夹
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

class DebugVGG16(nn.Module):
    """加载本地预训练的VGG16模型"""
    def __init__(self, num_classes=200):
        super().__init__()
        # 初始化标准VGG16结构（不加载预训练权重）
        # 我尝试过,这里加载weights=models.VGG16_Weights.IMAGENET1K_V1不会有性能提升,和None是一样的
        original_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) 
        
        # 特征提取层保持不变
        self.features = original_model.features
        
        # 动态计算分类器输入维度（适配64x64输入）
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.features(dummy)
            # in_features = features.view(1, -1).size(1) # 这行和下面那行只能留下一行
            in_features = features.view(-1).shape[0] # 这行是直接从train.py中抄过来的

        # 重建分类器层
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5), #这个放在别处可能也会影响性能，但是这里和训练时完全一致。我觉得可能影响不大
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5), #同上
            nn.Linear(4096, num_classes)
        )

        # 加载本地预训练权重
        pretrained_dict = torch.load("./train_vgg16/best_vgg16.pth", map_location="cuda:0")
        self.load_state_dict(pretrained_dict)
        # 加载权重后，打印缺失和意外的键
        missing, unexpected = self.load_state_dict(pretrained_dict, strict=False)
        print("缺失的键（未加载的权重）:", missing)
        print("意外的键（多余的权重）:", unexpected)
        print("成功加载本地VGG16权重")



    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# ---------------------------- CMA-ES优化器 ---------------------------
class CMAESBlock:
    """管理每个块的CMA-ES优化器"""
    def __init__(self, block_size):
        self.dim = block_size**2 * 3  # 这个好像是块中的像素点个数，一会打印一下？
        # 动态计算 lambda 和 mu
        self.lambda_ = int(4 + 3 * math.log(self.dim))  # 公式 λ = 4 + 3*ln(n)
        self.mu = max(1, self.lambda_ // 2)             # μ = λ/2（向下取整，至少为1）
        
        self.mean = np.zeros(self.dim)
        self.C = np.eye(self.dim)  # 协方差矩阵;初始化为单位矩阵
        self.sigma = cma_sigma
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.mu = cma_mu
        self.lambda_ = cma_lambda
        self.cc = 4 / (self.dim + 4)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.cmu = 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu)
        self.damps = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1))-1) + self.cc
        self.chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))


        # 新增权重定义
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i+1) for i in range(self.mu)])
        self.weights = self.weights / self.weights.sum()

        # 历史记录，后面画CMA-ES图有用
        self.sigma_history = []     # 记录sigma的演化

        #这是为了块内稀疏(进而体现为整体稀疏)而引入的
        # 新增稀疏性参数
        self.sparsity = 0.1             # 初始稀疏度（扰动前10%的像素）
        self.active_pixels = []         # 记录当前激活的像素索引



    def sample(self):
        """生成候选样本"""
        return [self.sigma * np.random.multivariate_normal(self.mean, self.C) 
                for _ in range(self.lambda_)]

    def update(self, parents):
        """更新参数"""
        
        if len(parents) != self.mu:
            raise ValueError("Parents数量必须等于mu")

        # 先前的更新均值策略，现已替换成加权
        # new_mean = np.mean(parents, axis=0) 

        # 加权均值计算
        weighted_mean = np.zeros_like(self.mean)
        for w, x in zip(self.weights, parents):
            weighted_mean += w * x
        new_mean = weighted_mean




        y = (new_mean - self.mean) / self.sigma # --这个是进化路径计算中的一项
        self.p_sigma = (1 - self.cc) * self.p_sigma + np.sqrt(self.cc*(2-self.cc)*self.mu) * y #更新进化路径(步长)
        sigma_norm = np.linalg.norm(self.p_sigma) / self.chi_n
        self.sigma *= np.exp((sigma_norm - 1) * self.cc / self.damps) # 更新步长
        # self.p_c = (1 - self.cc) * self.p_c + np.sqrt(self.cc*(2-self.cc)*self.mu) * y # 更新进化路径(协方差矩阵)
        # delta_C = np.outer(self.p_c, self.p_c) * self.c1 #--这个是更新协方差矩阵的第二项
        for x in parents:
            z = (x - self.mean) / self.sigma
            # delta_C += self.cmu * np.outer(z, z)
        # self.C = (1 - selff.c1 - self.cmu) * self.C + delta_C
        self.mean = new_mean.copy()
        # self.C = (self.C + self.C.T) / 2
        # 打印更新后的参数
        # print(f"均值变化范数: {np.linalg.norm(self.mean - old_mean):.4f}")
        # print(f"协方差矩阵迹: {np.trace(self.C):.4f}")

        # 记录历史状态,后面画CMA-ES图有用
        self.sigma_history.append(self.sigma)


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
            # print(f"Initial best score: {self.best_score}")
        elif score > self.best_score - self.min_delta:# 准确率越低越好
            self.counter += 1
            # print(f"Score {score} is less than best score {self.best_score} + min_delta {self.min_delta}. Counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at score {score} after {self.counter} iterations without improvement.")
        else:
            self.best_score = score
            self.counter = 0
            # print(f"New best score: {self.best_score}")
        return self.early_stop
# ---------------------------- 攻击类 ---------------------------
class SZOAttack:
    """稀疏区域优化的对抗攻击"""
    def __init__(self, model, image, label, block_size, device, epsilon=0.1,universal_perturbation = None,original_path=None):  # 新增epsilon参数
        self.device = image.device  # 继承输入图像的设备
        self.model = model.to(self.device)
        self.original_image = image.clone().detach().to(self.device) #由于在transform函数中已经做过归一化，这里也是归一化过的
        self.label = label.to(self.device) if isinstance(label, torch.Tensor) else torch.tensor(label, device=self.device)
        #20250308 这一版不知道怎么了，之前协方差矩阵也能更新且不用输出攻击报告的时候，label也不是张量(我记得是int)，也可以to.device(),
        #但是现在不行了，所以只能弄成tensor.后续又出现self.universal_perturbation 和 delta 不在同一个设备上，delta也要to.device().

        self.block_size = block_size
        self.epsilon = epsilon  # 关键：定义epsilon属性
        self.original_path = original_path  # 存储原始图像路径
        
        # 确保所有初始化张量在GPU
        self.H, self.W = image.shape[-2:]  # 使用最后两个维度（H,W）
        self.num_blocks_h = (self.H + block_size - 1) // block_size
        self.num_blocks_w = (self.W + block_size - 1) // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w
        
        # 使用GPU张量初始化概率
        self.prob = torch.ones(self.total_blocks, device=self.device) / self.total_blocks
        
        # 预分配GPU显存空间
        self.best_adv = torch.empty_like(self.original_image, device=self.device)
        self.best_loss = -np.inf
        self.target_class = None
        
        # 其他初始化代码...
        self.global_acc_history = []  # 新增：初始化历史记录列表
        self.eval_interval = 1        # 每次迭代评估一次全局准确率

        # 这个定义之前没写也可以运行，不知道为什么，还是先补上了
        self.cma_blocks = {bid: CMAESBlock(block_size) for bid in range(self.total_blocks)}


        # for bid in range(self.total_blocks):
        #     block_pixels = block_size**2 * 3  # 计算块像素数 (RGB)
        #     self.cma_blocks[bid] = CMAESBlock(block_pixels)
        #     print(f"块 {bid}: 像素数={block_pixels}, lambda={self.cma_blocks[bid].lambda_}, mu={self.cma_blocks[bid].mu}")
        # 初始化时,接收外部传入的全局扰动
        if universal_perturbation is None:
            self.universal_perturbation = torch.zeros_like(image.unsqueeze(0), device=device)
        else:
            self.universal_perturbation = universal_perturbation.clone().to(self.device)
        ##上面这行代码的原因: [1,3,H,W],设置为4维,这样才能与image匹配(images.shape = [64, 3, 64, 64]  # Batch size=64)
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)  # 初始化早停策略

        # 条件化初始化可视化组件
        if debug_mode:
            self.visualizer = AttackVisualizer(
                H=self.H,  # 正确传递参数
                W=self.W,
                block_size=self.block_size,
                device=self.device
            )
            # self.logger = CMAESLogger(
            #     total_blocks=self.total_blocks,  # 参数名修正log_block_selection
            #     device=self.device
            # )
        else:
            self.visualizer = None
            # self.logger = None


    # def _log_iteration(self, class_id, iteration, selected_blocks, delta):
    #     """记录单次迭代的所有信息:包含对self.visualizer.log_block_selection和self.visualizer.plot_perturbation的调用"""
    #     if not debug_mode:
    #         return
    #     # 块选择日志
    #     self.visualizer.log_block_selection(
    #         class_id=class_id,
    #         iteration=iteration,
    #         selected_blocks=selected_blocks,
    #         prob_dist=self.prob.cpu()
    #     )
        
    #     # CMA-ES状态记录
    #     for bid in selected_blocks:
    #         cma_state = {
    #             'mean': self.cma_blocks[bid].mean,
    #             'C': self.cma_blocks[bid].C,
    #             'sigma': self.cma_blocks[bid].sigma,
    #             'fitness': self.cma_blocks[bid].best_fitness
    #         }
            
    #     # 扰动可视化
    #     self.visualizer.plot_perturbation(
    #         delta=delta.detach().cpu(),
    #         universal_pert=self.universal_perturbation.detach().cpu(),
    #         iteration=iteration
    #     )
    
    def block_projection(self, delta, bid): #return delta
        """全GPU化的块投影"""
        num_blocks_per_row = (self.W + self.block_size - 1) // self.block_size
        h_start = (bid // num_blocks_per_row) * self.block_size
        w_start = (bid % num_blocks_per_row) * self.block_size
        
        # 使用GPU加速的切片操作
        h_slice = slice(max(0, h_start), min(h_start+self.block_size, self.H))
        w_slice = slice(max(0, w_start), min(w_start+self.block_size, self.W))

        h_pixels = h_slice.stop - h_slice.start
        w_pixels = w_slice.stop - w_slice.start
        # print(f"h_pixels: ", h_pixels) #打印出来是block_size
        # print(f"w_pixels: ", w_pixels) #打印出来也是block_size

        # 获取当前块的CMA参数
        cma_block = self.cma_blocks[bid]
        
        # 生成全扰动向量（未稀疏化）
        full_perturbation = cma_block.sigma * np.random.multivariate_normal(
            cma_block.mean, cma_block.C
        )
        
        # (硬稀疏)稀疏化：选择影响力最大的前k个像素
        # perturbation_flat = full_perturbation.reshape(-1)
        k = int(cma_block.sparsity * len(full_perturbation))  # 扰动像素数量
        top_indices = np.argsort(np.abs(full_perturbation))[-k:]  # 选择绝对值最大的k个
        
        # 生成稀疏扰动矩阵
        sparse_perturbation = np.zeros(3 * h_pixels * w_pixels)  
        sparse_perturbation[top_indices] = full_perturbation[top_indices]
        sparse_perturbation = sparse_perturbation.reshape(3, h_pixels, w_pixels)
        
        # 应用扰动到delta
        delta[:, h_slice, w_slice] += torch.tensor(sparse_perturbation, 
                                                device=self.device)
        return delta

    def sample_perturbations(self): # return delta.to(self.device), selected_blocks
        """批量生成GPU扰动"""
        selected_blocks = torch.multinomial(self.prob, num_samples, replacement=True)
        valid_bids = selected_blocks.unique().cpu().numpy()  # 仅CPU用于索引计算
        
        # 预分配GPU显存
        delta = torch.zeros_like(self.original_image, device=self.device)
        
        # 并行化块处理
        for bid in valid_bids:
            if bid >= self.total_blocks:
                continue
            delta = self.block_projection(delta, bid)
        
        return delta.to(self.device), selected_blocks
    
    
    def evaluate(self, adv_images): # return losses
        """评估对抗样本"""
        with torch.no_grad():
            logits = self.model(adv_images)
            
            if self.target_class is None:
                # 非目标攻击：最大化原始类别损失
                target = torch.full((logits.size(0),), self.label, device=self.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)
            else:
                 # 目标攻击：最小化目标类别损失
                target = torch.full((logits.size(0),), self.target_class, device=self.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target)
            # 添加正则化项（确保系数合理）
            sparsity_penalty = torch.norm(adv_images - self.original_image, p=1)
            losses += lambda_sparsity * sparsity_penalty
            
            # 打印调试信息
            # print(f"平均损失: {losses.mean().item():.4f}, 正则化项: {sparsity_penalty.item():.4f}")
            return losses

    def update_prob(self, blocks, losses):
        """更新选择概率"""
        success_blocks = blocks[losses.argsort(descending=True)[:mu]]
        self.prob = torch.zeros_like(self.prob)
        unique_blocks, counts = torch.unique(success_blocks, return_counts=True)
        # 打印出每次更新的概率信息
        print(f"\nUpdating probability with success blocks: {success_blocks[:10]}")  # 打印前10个成功的块
        for b, cnt in zip(unique_blocks, counts):
            self.prob[b] = cnt.float()**1
        self.prob = (self.prob + 0.01) / (self.prob.sum() + 0.01 * self.prob.size(0))
        print(f"\nUpdated probabilities: {self.prob[:10]}")  # 打印前10个概率
        # 更新CMA参数
        for bid in unique_blocks.cpu().numpy():
            parents = [self.cma_blocks[bid].mean + self.cma_blocks[bid].sigma * 
                      np.random.multivariate_normal(np.zeros(self.cma_blocks[bid].dim), self.cma_blocks[bid].C)
                      for _ in range(self.cma_blocks[bid].mu)]
            self.cma_blocks[bid].update(parents)

        # 1.3.1版本新增：动态调整每个块的稀疏度
        for bid in unique_blocks.cpu().numpy():
            block = self.cma_blocks[bid]
            if block.sparsity > 0.05:  # 最低稀疏度5%
                # 如果块表现好，降低稀疏度（允许更多像素被扰动）
                block.sparsity *= 0.95 if losses.mean() < self.best_loss else 1.1



    def attack_batch(self, images, labels):
        """全GPU批量攻击"""
        # 确保输入数据在GPU
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # 预分配结果显存
        adv_images = torch.empty_like(images, device=self.device)
        
        # 使用向量化操作替代多线程
        for i in range(images.shape[0]):
            # attacker = SZOAttack(self.model, images[i], labels[i], self.block_size,device=self.device,epsilon=self.epsilon) # 传递全局参数)
            # 应用当前的全局扰动
            # perturbed_img = torch.clamp(images[i] + self.universal_perturbation, 0, 1)
            perturbed_img = images[i] + self.universal_perturbation

            adv_images[i] = perturbed_img.squeeze(0)  # 移除批次维度
            
            # 显存优化
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
        return adv_images



    def attack(self, val_loader, device, enable_eval=True, show_progress=True):
        """最核心的函数：全GPU化攻击流程 (新增全局准确率记录功能)"""
        progress_bar = tqdm(range(max_iter), desc="🔥GPU攻击进程") if show_progress else range(max_iter)
        
        for t in progress_bar:
            # 1. 生成扰动并更新通用扰动
            delta, blocks = self.sample_perturbations()
            delta = delta.unsqueeze(0).to(self.device)  # 确保 delta 在 GPU 上

            # 2.更新全局扰动
            self.universal_perturbation = torch.clamp(
                self.universal_perturbation + delta,   # torch.clamp:对张量进行元素级的范围限制，每个元素都不能小于第二个参数，也不能大于第三个参数
                -self.epsilon,  # 扰动幅度约束
                self.epsilon
            )
            if debug_mode:
                # === 新增可视化调用点1：扰动生成后 ===
                self.visualizer.plot_perturbation(
                    cid=self.label.item(),
                    delta=delta.detach().cpu(),
                    universal_pert=self.universal_perturbation.detach().cpu(),
                    iteration=t
                )
            # print(f"\nIteration {t}, Epsilon: {self.epsilon}, Perturbation Norm: {torch.norm(self.universal_perturbation).item()}")

            # 3. 生成对抗样本
            adv_images = self.original_image.to(self.device) + self.universal_perturbation # 这里的adv_images是一个四维张量

                # ==== 新增调试代码 ====
            if epsilon == 0:
                # 数值一致性检查
                if not torch.allclose(self.original_image, adv_images, atol=1e-6):
                    print("⚠️ 零扰动下数据不一致！最大差异:", 
                        torch.max(torch.abs(self.original_image - adv_images)).item())
                
                # 数据范围检查
                print("原始数据范围: [{:.3f}, {:.3f}]".format(
                    self.original_image.min(), self.original_image.max()))
                print("扰动后范围: [{:.3f}, {:.3f}]".format(
                    adv_images.min(), adv_images.max()))
                
                
                # ==== 调试结束 ====
            # ==== 新增：每次迭代都生成报告 ====
            with torch.no_grad():
                outputs = self.model(adv_images)
                adv_pred = outputs.argmax().item()
                
            # 保存当前迭代的报告（包含迭代次数）
            self.save_attack_report(
                orig_img=self.original_image,
                adv_img=adv_images.squeeze(),
                true_label=self.label.item(),   #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                adv_label=adv_pred,
                original_path=self.original_path,
                iteration=t  # 新增迭代参数
            )

            # 4. 调用 evaluate 函数计算损失值 && 更新块选择概率
            losses = self.evaluate(adv_images)
            self.update_prob(blocks, losses)
            
            if debug_mode:
                # === 新增可视化调用点2,3：概率更新后 ===
                self.visualizer.log_block_heatmap(
                    cid=self.label.item(), #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                    iteration=t,
                    selected_blocks=blocks.cpu().numpy()
                )
                self.visualizer.log_block_distribution(
                    class_id=self.label.item(), #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                    iteration=t,
                    prob_dist=self.prob.cpu()
                )
            
                # === 新增可视化调用点4：CMA-ES更新 ===
                for bid in blocks.unique().cpu().numpy():
                    self.visualizer.visualize_cma_es(
                        cid=self.label.item(), #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                        bid = bid,
                        cma_state={
                            'mean': self.cma_blocks[bid].mean,
                            'C': self.cma_blocks[bid].C,
                            'sigma': self.cma_blocks[bid].sigma,  # 新增当前sigma
                            'sigma_history': self.cma_blocks[bid].sigma_history
                        },
                        iteration=t
                    )
            # 5. 计算当前最佳损失值
            with torch.no_grad():
                outputs = self.model(adv_images)
                loss = nn.CrossEntropyLoss()(outputs, self.label.view(-1))
                loss += lambda_sparsity * torch.norm(delta, p=1)
            
            # 6. 更新最佳结果
            if loss > self.best_loss:
                self.best_loss = loss
                self.best_adv = adv_images

            
            # 7. 记录全局准确率
            if enable_eval and (t % self.eval_interval == 0):
                attacked_val_images = []
                attacked_val_labels = []
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    adv_images = self.attack_batch(images, labels)
                    attacked_val_images.append(adv_images)
                    attacked_val_labels.append(labels)
                    # ------------------调试begin: 逐个图像处理(调试结果:为攻击情况下准确率过低的原因不在这)----------------
                    # for i in range(images.shape[0]):
                    #     perturbed_img = torch.clamp(images[i] + self.universal_perturbation, 0, 1)  # 添加全局扰动
                    #     attacked_val_images.append(perturbed_img)  # 添加扰动后的图像，并保持批次维度
                    #     attacked_val_labels.append(labels[i].unsqueeze(0))  # 添加标签，并保持批次维度
                    # ------------------调试end: 逐个图像处理----------------

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

            # 8. 更新进度条
            if enable_eval:
                progress_bar.set_postfix(
                    Global_Acc=f"{global_acc:.2f}%" if enable_eval else "N/A", 
                    Current_Acc=f"{current_acc:.2f}%", 
                    Loss=f"{loss.item():.4f}"
                )
        
        # 确保 global_acc_history 长度足够
        if len(self.global_acc_history) < max_iter:
            self.global_acc_history.extend([self.global_acc_history[-1]] * (max_iter - len(self.global_acc_history)))
        
        return self.best_adv.squeeze(0)

    def save_attack_report(self, orig_img, adv_img, true_label, adv_label, original_path, iteration):
        """保存攻击效果报告图（按类别组织目录结构）"""
        # 解析原始路径获取数据集类型和类别信息
        path_parts = original_path.split(os.sep)
        
        # 定位关键路径节点（假设路径结构为: .../tiny-imagenet-200/[train或val]/class_folder/images/xxx.JPEG）
        try:
            dataset_type = path_parts[path_parts.index('tiny-imagenet-200') + 1]  # 获取train或val
            class_folder = path_parts[path_parts.index(dataset_type) + 1]        # 获取类别文件夹名称
        except ValueError:
            dataset_type = "unknown_dataset"
            class_folder = "unknown_class"

        # 构建保存路径（格式: attacked_images/[train|val]/[class_folder]/）
        base_path = os.path.join("attacked_images", dataset_type, class_folder)
        os.makedirs(base_path, exist_ok=True)

        # 生成带迭代次数的文件名
        filename = os.path.basename(original_path).split('.')[0]
        save_path = os.path.join(base_path, f"{filename}_iter{iteration}_report.png")

        # 以下绘图代码保持不变...
        orig_img_np = orig_img.cpu().numpy().transpose(1, 2, 0)
        adv_img_np = adv_img.cpu().numpy().transpose(1, 2, 0)
        
        # 计算噪声（基于归一化后的张量）
        noise = (adv_img - orig_img).abs().sum(0, keepdim=True)
        noise_norm = noise.squeeze().cpu().numpy()
        noise_norm = (noise_norm - noise_norm.min()) / (noise_norm.max() - noise_norm.min() + 1e-8)

        # 绘制三图合一报告
        plt.figure(figsize=(15, 5))
        
        # 原始图像（归一化后）
        plt.subplot(1, 3, 1)
        plt.imshow(orig_img_np)
        plt.title(f"Original\nLabel: {true_label}")
        plt.axis('off')

        # 对抗样本（归一化后，未裁剪）
        plt.subplot(1, 3, 2)
        plt.imshow(adv_img_np)
        plt.title(f"Adversarial\nPred: {adv_label}")
        plt.axis('off')

        # 噪声热力图
        plt.subplot(1, 3, 3)
        heatmap = plt.imshow(noise_norm, cmap='viridis_r', vmin=0, vmax=0.3)
        plt.colorbar(heatmap, fraction=0.046, pad=0.04)
        plt.title("Perturbation Heatmap")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
class AttackVisualizer:
    """与SZOAttack尺寸完全兼容的可视化系统"""
    def __init__(self, H, W, block_size, device):
        self.H = H
        self.W = W
        self.block_size = block_size
        self.device = device
        self._init_filesystem()
        
    def _init_filesystem(self):
        """创建完整的日志目录体系,在类初始化阶段使用"""
        # 全局目录
        os.makedirs("attack_logs/perturbations", exist_ok=True)
        
        # 类专属目录（按最大可能类别数预创建）
        for cid in range(num_test_classes):  # 假设最多200个类别
            class_dirs = [
                f"attack_logs/class_{cid}/block_selections",
                f"attack_logs/class_{cid}/cma_es",
                f"attack_logs/perturbations/class_{cid}"
            ]
            for d in class_dirs:
                os.makedirs(d, exist_ok=True)

    def log_block_heatmap(self, cid, iteration, selected_blocks):
        """热力图专用方法（保留第一个版本核心逻辑）,此方法在def attack中被直接调用"""
        grid_size = self.H // self.block_size
        heatmap = torch.zeros((grid_size, grid_size), device='cpu')

        # 转换块ID到坐标
        for bid in selected_blocks:
            row = bid // grid_size
            col = bid % grid_size
            heatmap[row, col] += 1

        # 生成图像
        plt.imshow(heatmap.numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Class {cid} Block Heatmap\nIter {iteration}")
        plt.savefig(f"attack_logs/class_{cid}/block_selections/iter_{iteration:04d}.png")
        plt.close()

    def log_block_distribution(self, class_id, iteration, prob_dist):
        """概率分布专用方法（整合第二个版本特性）,此方法在def attack中被直接调用"""
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(prob_dist)), prob_dist.numpy(), alpha=0.7)
        plt.xlabel('Block ID')
        plt.ylabel('Selection Probability')
        plt.title(f"Class {class_id} Probability Distribution\nIter {iteration}")
        plt.savefig(f"attack_logs/class_{class_id}/block_selections/probs_iter_{iteration:04d}.png")
        plt.close()

    def visualize_cma_es(self, cid, bid, cma_state, iteration):
        """可视化CMA-ES搜索过程,此方法在def attack中被直接调用"""
        # 创建类专属目录
        class_dir = f"attack_logs/class_{cid}"
        cma_es_dir = os.path.join(class_dir, "cma_es")
        os.makedirs(cma_es_dir, exist_ok=True)

        # 创建可视化图形
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # === 左图：协方差椭圆 ===
        current_sigma = cma_state['sigma']
        cov = cma_state['C'] * (current_sigma ** 2)  # 当前协方差
        
        # 特征分解（简化）
        radius = 2 * current_sigma  # 协方差矩阵固定为单位阵
        w, v = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
        ellipse = Ellipse(
            xy=cma_state['mean'],
            width=2*np.sqrt(w[0]),
            height=2*np.sqrt(w[1]),
            angle=angle,
            fc='none', 
            ec='red'
        )
        axs[0].add_patch(ellipse)
        # 动态坐标范围
        max_radius = max(1.5, radius*1.2)
        axs[0].set_xlim(-max_radius, max_radius)
        axs[0].set_ylim(-max_radius, max_radius)
        axs[0].set_title(f"Block {bid} Search Space\nσ={cma_state['sigma']:.3f}")
        
        # 参数演化曲线
        axs[1].plot(cma_state['sigma_history'], label='Sigma')
        axs[1].legend()
        axs[1].set_title("Parameter Evolution")
        
        # 保存并关闭
        plt.savefig(os.path.join(cma_es_dir, f"block_{bid}_iter_{iteration:04d}.png"), dpi=150)
        plt.close()
    def plot_perturbation(self, cid, delta, universal_pert, iteration):
        """扰动可视化对比"""
        class_dir = f"attack_logs/perturbations/class_{cid}"
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 当前迭代扰动
        axs[0].imshow(delta.squeeze().permute(1,2,0).numpy() * 3 + 0.5)
        axs[0].set_title(f"Class {cid} Iter {iteration} Delta")
        
        # 累积扰动
        axs[1].imshow(universal_pert.squeeze().permute(1,2,0).numpy() * 3 + 0.5)
        axs[1].set_title(f"Class {cid} Accumulated Pert")
        
        # 扰动幅度热力图
        heatmap = torch.norm(universal_pert.squeeze(), dim=0)
        axs[2].imshow(heatmap, cmap='inferno')
        axs[2].set_title(f"Class {cid} Pert Magnitude")
        
        plt.savefig(os.path.join(class_dir, f"iter_{iteration:04d}.png"))
        plt.close()


# 这个函数被暂时弃用了，因为它的准确率计算方式不对
# 原因是这里面多进行了一次标准化。标准化只能进行一次，不能多做！！！
# 还有一个原因：难道计算准确率应该使用全局函数，而不要让model传入一个类再使用类内函数计算吗？这是为什么？
# ============================工具函数===================
def evaluate_global_accuracy(model, val_loader, device):
    """评估带/不带扰动的全局准确率,其中：
    torch.no_grad() : 禁用梯度计算，主要用于 推理(inference)或 评估模型时。它能够 减少内存占用 和 加速计算，因为在评估阶段，我们并不需要计算梯度。
    torch.cuda.amp.autocast() : 自动混合精度。使用fp16加速,降低显存占用"""
    model.eval()
    correct = 0
    total = 0
    # with torch.no_grad(), torch.cuda.amp.autocast(): 
    with torch.no_grad(): #因为label 
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # 在 outputs 张量的第 1 个维度（即类别维度）上找到最大值，并返回最大值及其对应的索引。这里使用了 Python 的元组解包语法，_ 表示我们不关心最大值具体是多少，只关心最大值所在的索引
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100




# ---------------------------- 主函数 ---------------------------
def main():
    # 初始化设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./tiny-imagenet-200"
    # set_seed()
    
    # 数据加载val_loader 在后续代码中多次被使用，例如在评估模型的原始准确率和进行对抗攻击时，都会从 val_loader 中获取数据。
    # val_loader中的数据是已经归一化的
    val_loader = get_tinyimagenet_loader(data_root, train=False) 
    
    # 加载模型
    model = DebugVGG16(num_classes=200).to(device) #模型移动到GPU
    # 在进行推理时，确保调用 model.eval()！
    model.eval()
    #=================debug : 验证特征图尺寸(开始)===========
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print("输出张量形状:", output.shape)  # 应输出 [1, 200]
        features = model.features(dummy_input)
        print(f"特征图尺寸: {features.shape}")  # 应输出如 [1, 512, 2, 2]
        flattened = torch.flatten(features, 1)
        print(f"展平后维度: {flattened.shape}")   # 应如 [1, 512*2*2=2048]
    # =================debug : 验证特征图尺寸(结束)=========== 
    # 验证原始准确率
    orig_acc = evaluate_global_accuracy(model, val_loader, device) # 这里的原始准确率已经正常了,侧面说明这个函数是没有问题的。如果不正常可能是归一化次数超过一次或者做了不必要的数据增强导致
    
    print(f"\n原始模型准确率: {orig_acc:.2f}%")
    
    # 按类别抽样测试（修改标签处理部分）
    # 抽样得到的样本将存储在 class_samples 字典中，img 是该类别的图像张量，lbl 是对应的标签张量。
    # class_samples存放的是原始的图像张量和标签
    class_samples = {}
    dataset = val_loader.dataset
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        lbl_value = lbl
        if lbl_value not in class_samples:
            # 存储图像路径信息
            img_pth = dataset.imgs[idx][0]
            label_tensor = torch.tensor(lbl, dtype=torch.long)  # 转换为张量
            class_samples[lbl_value] = (img, label_tensor, img_pth)
        if len(class_samples) >= num_test_classes:
            break
    
    # 执行攻击
    attack_success = 0
    total_l0, total_l1 = 0, 0.0
    # 在main函数中定义全局扰动,注意这个扰动是跨类别的,这样在不同类别上的扰动效果才能相互影响
    universal_perturbation = torch.zeros((1,3,64,64), device=device)  # 初始化为四维
    # 这个数据结构:存储所有类别的攻击历史
    all_class_history = {
        # 格式：{class_id: {'global_acc': [], 'iterations': []}, ...}
    }

    test_progress = tqdm(class_samples.items(), desc="攻击进度")
    
    for class_id, (image, label,img_path) in test_progress:  # 此处label已经是张量
        # 这两行是当前版本新加的:将当前图像和标签移动到GPU
        image = image.to(device)
        label = label.to(device)
        attacker = SZOAttack(
            model=model,
            image=image, # 原始图像
            label=label, # 原始标签(这里是张量)
            block_size=block_size,
            original_path=img_path,
            device=device,
            epsilon=epsilon,  # 传递全局参数,不然会使用默认值0.1
            universal_perturbation=universal_perturbation  # 传入共享扰动
        )
        print("Visualizer exists:", hasattr(attacker, 'visualizer'))  # 应输出True
        # print("Logger exists:", hasattr(attacker, 'logger'))          # 应输出True
        adv_img = attacker.attack(val_loader, device)
        # 记录当前类别的攻击历史
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
            plt.close()  # 避免内存泄漏


        universal_perturbation = attacker.universal_perturbation  # 更新全局扰动
        
        # 评估攻击效果（修改比较逻辑）
        with torch.no_grad():
            logits = model(adv_img.unsqueeze(0))
            pred = logits.argmax().item()
            attack_success += int(pred != label.item())  # 用.item()获取整型值比较
        
        # 计算扰动指标
        delta = (adv_img - image).abs().sum(0)
        total_l0 += (delta > 0.005).sum().item()
        total_l1 += delta.sum().item()
        
        # 更新进度
        current_sr = attack_success / (list(class_samples.keys()).index(class_id) + 1) * 100
        test_progress.set_postfix(Success=f"{current_sr:.1f}%")
    
    # 打印结果
    print(f"\n最终攻击成功率: {attack_success/len(class_samples)*100:.1f}%")
    print(f"平均L0扰动: {total_l0/len(class_samples):.1f}, 平均L1扰动: {total_l1/len(class_samples):.1f}")
    
    # 只加载一个类的绘图逻辑(暂时注释掉)
    # if attacker.global_acc_history:
    #     plt.figure(figsize=(10, 5))
    #     x_axis = np.arange(0, len(attacker.global_acc_history)*attacker.eval_interval, attacker.eval_interval)
    #     plt.plot(x_axis, attacker.global_acc_history, marker='o', color='red', label='Global Acc (Perturbed)')
        
    #     # 绘制原始准确率作为对比基线
    #     plt.axhline(y=orig_acc, color='blue', linestyle='--', label='Original Accuracy')
        
    #     plt.title("Global Accuracy Under Universal Adversarial Attack")
    #     plt.xlabel("Attack Iteration")
    #     plt.ylabel("Accuracy (%)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("global_acc_trend.png")
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_class_history)))

    for (class_id, data), color in zip(all_class_history.items(), colors):


        # 填充数据，使得 global_acc 和 iterations 的长度相同
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_classes_accuracy_trend.png", bbox_inches='tight')


    with open('attack_history.json', 'w') as f:
        json.dump(all_class_history, f)

if __name__ == "__main__":
    main()

'''
3.26晚运行结果:
arly stopping triggered at score 50.67358547051193 after 10 iterations without improvement.
Early stopping at iteration 22 due to no improvement in global accuracy.
🔥GPU攻击进程:  44%|██▏  | 22/50 [03:31<04:29,  9.63s/it, Current_Acc=50.67%, Global_Acc=50.79%, Loss=49.5213]
攻击进度: 100%|█████████████████████████████████████████████| 20/20 [1:23:06<00:00, 249.34s/it, Success=30.0%]

最终攻击成功率: 30.0%
平均L0扰动: 3827.2, 平均L1扰动: 1481.3
'''