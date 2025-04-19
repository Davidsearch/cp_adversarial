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
epsilon = 0.25       # 总扰动预算
# 块选择策略
block_size = 8     # 块尺寸
num_samples = 16     # 每轮采样块数
block_mu = 1         # 块选择策略中成功块的个数(老师的要求这个最终可能要设置成1 ?)
max_iter = 10      # 增加迭代次数以观察趋势,这个不应该设的太小(比如3)
num_test_classes = 200  # 测试类别数（从200类中随机选择）
target_class = None # 当前是无目标攻击 
lambda_sparsity = 0.3  # 稀疏性正则化系数

debug_mode = 0  # 1启用日志和报告，0禁用（设置为0时执行原版逻辑）

# --- CMA-ES 参数 ---


# cma_lambda = 10     # 种群大小[该参数在当前版本代码中未使用,代码中用的是4 + 3ln(n)]
# cma_mu = 5          # 父代数量[该参数未使用,代码中使用和n有关的函数]
cma_sigma = 0.2     # 初始步长
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
# 通过一个参数的调整，可以加载1个样本，也可以是某个百分比的样本（比如10%、20%等）
def get_tinyimagenet_loader(data_dir, batch_size=64, train=False):
    """加载Tiny ImageNet数据集（每个类别10%数据）"""
    transform = val_transform if not train else train_transform
    
    dataset_path = os.path.join(data_dir, 'train' if train else 'val')
    
    # 清理检查点文件夹
    checkpoint_path = os.path.join(dataset_path, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    
    # 加载完整数据集
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    # 按类别分层采样10%
    from collections import defaultdict
    import random
    random.seed(42)  # 固定随机种子保证可重复性
    
    # 创建类别->索引的映射
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
    
    # 计算每个类别采样数量（至少1个）
    sampled_indices = []
    for label, indices in class_indices.items():
        n_total = len(indices)
        n_samples = max(1, int(0.1 * n_total))  # 关键修改点：10%采样
        sampled_indices.extend(random.sample(indices, n_samples))
    
    # 创建采样后的子集
    sampled_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
    
    return DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        shuffle=False,         # 保持原始shuffle参数
        num_workers=4,
        pin_memory=True
    )

# 新增Thompson Sampling概率更新类，这个在SZOAttack中会化成一个ts_updater对象
class ThompsonSamplingUpdater:
    def __init__(self, total_blocks, device):
        self.alpha = torch.ones(total_blocks, device=device) + 1e-6  # 成功计数
        self.beta = torch.ones(total_blocks, device=device)   # 失败计数
        self.total_blocks = total_blocks
        self.device = device

    def update(self, selected_blocks, successes): #更新块内扰动
        """更新Beta分布参数"""
        for bid in selected_blocks.unique():
            mask = (selected_blocks == bid)
            success_count = successes[mask].sum().item()
            fail_count = mask.sum().item() - success_count
            
            self.alpha[bid] += success_count
            self.beta[bid] += fail_count

    def sample_probs(self):
        """从Beta分布采样概率"""
        return torch.distributions.Beta(self.alpha, self.beta).sample()


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
        pretrained_dict = torch.load("../train_vgg16/best_vgg16_1.pth", map_location="cuda:0")
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
        self.best_fitness = -np.inf  # 新增：跟踪块的最佳适应度
        # 动态计算 lambda 和 mu
        self.lambda_ = int(4 + 3 * math.log(self.dim))  # 公式 λ = 4 + 3*ln(n) pdf公式(48)
        self.mu = max(1, self.lambda_ // 2)             # μ = λ/2（"//"是整除的意思，向下取整，至少为1）,
        # self.lambda_ = cma_lambda  # 不再动态计算
        # self.mu = cma_mu           # 直接使用全局参数
        self.mean = np.zeros(self.dim)
        self.C = np.eye(self.dim)  # 协方差矩阵;初始化为单位矩阵
        self.sigma = cma_sigma #这个是硬设置的，可以根据实际情况进行调整
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)

        # 权重计算（PDF公式49-53）
        raw_weights = [math.log((self.lambda_ + 1)/2) - math.log(i+1) for i in range(self.lambda_)]
        positive_weights = raw_weights[:self.mu]
        negative_weights = raw_weights[self.mu:]

        sum_pos = sum(abs(w) for w in positive_weights)
        sum_neg = sum(abs(w) for w in negative_weights)
        positive_weights = [w/sum_pos for w in positive_weights]

        # 计算μ_eff（PDF符号说明第8段）
        self.mueff = (sum(positive_weights)**2) / sum(w**2 for w in positive_weights)

        # 计算负权重缩放因子（PDF公式50-53）
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)  # 临时值，后续修正
        self.cmu = 0.25  # 临时值
        alpha_mu_neg = 1 + self.c1 / self.cmu
        mueff_neg = (sum(negative_weights)**2) / sum(w**2 for w in negative_weights)
        alpha_mueff_neg = 1 + 2 * mueff_neg / (self.mueff + 2)
        alpha_posdef_neg = (1 - self.c1 - self.cmu) / (self.dim * self.cmu)
        scale_neg = min(alpha_mu_neg, alpha_mueff_neg, alpha_posdef_neg) / sum_neg
        negative_weights = [w * scale_neg for w in negative_weights]

        self.weights = np.array(positive_weights + negative_weights)
        # self.mu = cma_mu
        # self.lambda_ = cma_lambda #先把这里注释掉，不采用人为设置的硬参数
        # 参数计算（PDF公式55-58）
        self.cc = (4 + self.mueff/self.dim) / (self.dim + 4 + 2*self.mueff/self.dim)  # 公式56
        alpha_cov = 2
        temp_cmu = alpha_cov * (0.25 + self.mueff + 1/self.mueff - 2) / ((self.dim + 2)**2 + alpha_cov * self.mueff / 2)
        # self.c1 = min(1 - self.cmu, temp_cmu)
        self.cmu = min(1 - self.c1, temp_cmu)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)  # 公式55
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.dim+1)) -1) + self.cs
                # 新增参数历史记录
        self.c1_history = []
        self.cmu_history = []
        self.p_sigma_history = []
        self.p_c_history = []
        self.C_history = []  # 协方差矩阵历史
        self.sigma_history = []     # 记录sigma的演化

        # 新增权重定义
        '''
        用户的代码中self.weights是取前mu个，用np.log(mu + 0.5) - np.log(i+1)，然后归一化。但PDF中
        的公式49是w'_i = ln((λ+1)/2) - ln(i)，对于i=1到λ。用户这里用的是mu而不是λ，这可能有问题。
        因为PDF中的权重是针对整个λ个样本的，而用户只取了前mu个，这显然不对。正确的做法应该是为所有λ
        样本生成权重，然后根据正负进行调整，比如公式49-53。用户的代码中只生成了mu个正权重，而PDF中可
        能还有负权重，所以这里明显有误。
        '''


        #注意，pdf中特别提示53式(权重设置)一定不能随便改，但是我怎么觉得我的代码中的设置和pdf不一致？
        # 历史记录，后面画CMA-ES图有用
        self.chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))


        #这是为了块内稀疏(进而体现为整体稀疏)而引入的
        # 新增稀疏性参数
        self.sparsity = 0.1             # 初始稀疏度（扰动前10%的像素）
        self.active_pixels = []         # 记录当前激活的像素索引

    def sample(self):
        self.samples = [
            self.mean + self.sigma * np.random.multivariate_normal(np.zeros(self.dim), self.C)
            for _ in range(self.lambda_)
        ]
        return self.samples
    def update(self, all_samples, fitness_values): #传进来的时候，all_samples是list，fitness_values是所有样本的适应度np.array。这个函数的bug应该是调完了
        """修正后的更新函数"""
        # === 输入预处理 ===
        all_samples = np.array(all_samples)
        fitness_values = np.array(fitness_values)
        
        # === 动态调整有效样本数 ===
        valid_samples = []
        valid_fitness = []
        for s, f in zip(all_samples, fitness_values):
            if not np.isnan(f) and s.shape == (self.dim,):
                valid_samples.append(s)
                valid_fitness.append(f)
        if len(valid_samples) < 1:
            return  # 无有效样本时跳过更新
        
        # === 动态计算实际mu值 ===
        actual_mu = min(self.mu, len(valid_samples))
        # print("actual_mu:", actual_mu)
        sorted_indices = np.argsort(valid_fitness)[::-1][:actual_mu]  # 关键修复：限制索引范围
        
        # === 权重归一化 ===
        actual_weights = self.weights[:actual_mu]  # 截取有效权重
        actual_weights /= np.sum(actual_weights)   # 重新归一化
        
        # === 均值更新 ===
        y_k = [(x - self.mean)/self.sigma for x in valid_samples]
        y_w = sum(w * y_k[i] for w, i in zip(actual_weights, sorted_indices))
        self.mean += self.sigma * y_w
        
        # 进化路径更新（保留但不影响协方差矩阵）
        generation = len(self.sigma_history)  
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2/(self.dim+1)) * self.chi_n
        h_sigma = 1 if p_sigma_norm / np.sqrt(1 - (1-self.cs)**(2*(generation+1))) < threshold else 0
        
        self.p_sigma = (1 - self.cs) * self.p_sigma + np.sqrt(self.cs*(2-self.cs)*self.mueff) * y_w
        self.p_c = (1 - self.cc) * self.p_c + h_sigma * np.sqrt(self.cc*(2-self.cc)*self.mueff) * y_w
        
        # 步长更新（保留）
        sigma_norm = np.linalg.norm(self.p_sigma) / self.chi_n
        self.sigma *= np.exp((sigma_norm - 1) * self.cs / self.damps)
        # 注释掉所有协方差矩阵更新部分 --------------------------------------------------
        rank1_update = self.c1 * np.outer(self.p_c, self.p_c)  # 秩1更新（注释）
        rank_mu_update = np.zeros_like(self.C)  # 秩μ更新（注释）
        for w, i in zip(self.weights, sorted_indices):
            y = y_k[i]
            rank_mu_update += w * np.outer(y, y)
        rank_mu_update = self.cmu * rank_mu_update
        sum_weights = sum(self.weights) 
        self.C = (1 - self.c1 - self.cmu*sum_weights) * self.C + rank1_update + rank_mu_update  # 组合更新（注释）
        
        # 注释数值稳定性处理（因协方差矩阵固定）
        self.C = (self.C + self.C.T) / 2  # （注释）
        self.C = np.clip(self.C, 1e-8, None)  # （注释）
        
        # 均值更新（保留）
        self.mean = self.mean.copy()
        
        # 更新历史记录并打印参数
        self.sigma_history.append(self.sigma)
        # print(f"sigma: {self.sigma}")
        
        self.c1_history.append(self.c1)
        # print(f"c1: {self.c1}")
        
        self.cmu_history.append(self.cmu)
        # print(f"cmu: {self.cmu}")
        
        self.p_sigma_history.append(self.p_sigma.copy())
        # print(f"p_sigma: {self.p_sigma}")
        
        self.p_c_history.append(self.p_c.copy())
        # print(f"p_c: {self.p_c}")
        
        self.C_history.append(self.C.copy())
        # print(f"C: {self.C}")



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
        self.mu = block_mu
        self.original_image = image.clone().detach().to(self.device) #由于在transform函数中已经做过归一化，这里也是归一化过的
        self.label = label.to(self.device) if isinstance(label, torch.Tensor) else torch.tensor(label, device=self.device)
        #20250308 这一版不知道怎么了，之前协方差矩阵也能更新且不用输出攻击报告的时候，label也不是张量(我记得是int)，也可以to.device(),
        #但是现在不行了，所以只能弄成tensor.后续又出现self.universal_perturbation 和 delta 不在同一个设备上，delta也要to.device().
       # === 新增批量图像支持 ===
        # self.original_images_batch = images_batch.clone().detach().to(self.device)  # 形状 [B, C, H, W]
        # self.labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device)
        # self.original_paths = original_paths  # 新增路径列表支持

        self.deltas = []
        self.block_size = block_size
        self.epsilon = epsilon  # 关键：定义epsilon属性
        self.original_path = original_path  # 存储原始图像路径
        
        # 确保所有初始化张量在GPU
        self.H, self.W = image.shape[-2:]  # 使用最后两个维度（H,W）
        self.num_blocks_h = (self.H + block_size - 1) // block_size
        self.num_blocks_w = (self.W + block_size - 1) // block_size
        self.total_blocks = self.num_blocks_h * self.num_blocks_w
        
        # 使用GPU张量初始化概率
        # 下面这一行是Thompson sampling专门加的
        self.ts_updater = ThompsonSamplingUpdater(self.total_blocks, device)

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
            self.universal_perturbation = torch.zeros_like(image, device=image.device)
            # self.universal_perturbation = torch.zeros_like(image.unsqueeze(0), device=device)

        else:
            self.universal_perturbation = universal_perturbation.squeeze(0).to(self.device)
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
            # print("full_perturbation.shape",full_perturbation.shape)
            
            # (硬稀疏)稀疏化：选择影响力最大的前k个像素
            # perturbation_flat = full_perturbation.reshape(-1)
            k = int(cma_block.sparsity * len(full_perturbation))  # 扰动像素数量
            top_indices = np.argsort(np.abs(full_perturbation))[-k:]  # 选择绝对值最大的k个
            
            # 生成稀疏扰动矩阵
            sparse_perturbation = np.zeros(3 * h_pixels * w_pixels)  
            sparse_perturbation[top_indices] = full_perturbation[top_indices]
            sparse_perturbation = sparse_perturbation.reshape(3, self.block_size, self.block_size)  # 正确3D形状
    
            # 转换为Tensor并适配批次维度
            sparse_tensor = torch.tensor(sparse_perturbation, device=self.device).unsqueeze(0)
            # print("sparse_tensor.shape:",sparse_tensor.shape)
            # 关键修复：正确切片和维度对齐
            # delta形状: [batch_size, C, H, W]
            delta[:, :, h_slice, w_slice] += sparse_tensor 
            # cma_block.append(sparse_perturbation.copy())  # 这行压根就没有意义，留在这里为了提醒其他代码也删除
            
            return delta

    def sample_perturbations(self):
        """生成分块扰动（修复num_samples作用）"""
        # 1. 选择多个块（每轮选num_samples个不同的块）
        selected_blocks = torch.multinomial(self.prob, num_samples, replacement=False).cpu().numpy()
        # print("selected_blocks.shape",selected_blocks.shape)
        
        # 2. 初始化扰动张量（形状 [num_samples, C* H* W]）
        deltas = torch.zeros((num_samples, *self.original_image.shape), device=self.device)
        # print("初始化生成的扰动形状:", deltas.shape) #torch.Size([5, 3, 64, 64]) 
        # 3. 使用block_projection为每个样本生成对应块扰动
        for sample_idx in range(num_samples):
            bid = int(selected_blocks[sample_idx])  # 转换为Python整数
            
            # 创建当前样本的delta副本
            single_delta = deltas[sample_idx].unsqueeze(0)  # [1, 3, H, W]
            # print("single_delta.shape",single_delta.shape)
            # 应用块投影
            perturbed = self.block_projection(single_delta, bid)
            # print("perturbed.shape:",perturbed.shape)
            # 写回结果
            deltas[sample_idx] = perturbed.squeeze(0)
        
        return deltas, torch.tensor(selected_blocks, device=self.device)
        
    def evaluate(self, adv_images): # return losses
        """评估对抗样本"""
            # 强制输入为4D
        # if adv_images.dim() == 3:
        #     adv_images = adv_images.unsqueeze(0)
        # elif adv_images.dim() != 4:
        #     raise ValueError(f"评估输入必须为3D或4D，实际维度: {adv_images.shape}")
        """批量评估多个候选样本"""
        if adv_images.dim() not in [3, 4]:
            raise ValueError(f"输入维度错误: {adv_images.shape}")

        with torch.no_grad():
            logits = self.model(adv_images)
            
            if self.target_class is None:
                # 非目标攻击：最大化原始类别损失
                target = torch.full((logits.size(0),), self.label, device=self.device)
                losses = nn.CrossEntropyLoss(reduction='none')(logits, target)
            else:
                 # 目标攻击：最小化目标类别损失
                target = torch.full((logits.size(0),), self.target_class, device=self.device)
                losses = -nn.CrossEntropyLoss(reduction='none')(logits, target) #取负号转为最大化问题
            # 添加正则化项（确保系数合理）
            sparsity_penalty = torch.norm(adv_images - self.original_image, p=1)
            losses -= lambda_sparsity * sparsity_penalty 
            
            # update_prob# 打印调试信息
            # print(f"平均损失: {losses.mecma_blocksan().item():.4f}, 正则化项: {sparsity_penalty.item():.4f}")
            return losses #一个张量，包含了每个输入样本的损失值

    def update_prob(self, blocks, fitness_values):
        """更新块选择概率并执行CMA-ES参数更新
        
        参数:
            blocks: 一维张量，形状为(num_samples,)，元素为每个样本选择的块ID
            fitness_values: numpy数组，形状为(num_samples,)，每个样本的损失值
        """
            # 输入校验
        assert len(blocks) == num_samples, "块总数必须等于num_samples"
        assert len(fitness_values) == num_samples, "适应度值数量必须匹配"
        # 转换为CPU numpy数组
        blocks_np = blocks.cpu().numpy() if isinstance(blocks, torch.Tensor) else blocks
        assert blocks_np.ndim == 1, "blocks必须是一维数组"
        
        # 1. 定义成功标准（选择前mu个高损失样本）
        sorted_indices = np.argsort(fitness_values)[-self.mu:]
        success_blocks = blocks_np[sorted_indices]
        # 注：NumPy 数组本身是 CPU-only 的，无法直接在 GPU 上运行
    
        
        # 2. 更新Beta分布参数（Thompson Sampling核心）
        for bid in np.unique(success_blocks):
            mask = (blocks_np == bid)
            success_count = mask.sum()  # 该块被选为高损失块的次数
            fail_count = len(blocks) - success_count  # 未被选中的次数
            
            # 增量更新alpha/beta（避免覆盖历史信息）
            self.ts_updater.alpha[bid] += success_count
            self.ts_updater.beta[bid] += fail_count
        
        # 3. 从Beta分布采样新概率（替代原有权重计算）
        self.prob = self.ts_updater.sample_probs()
        
        # 4. CMA-ES更新（保留原有逻辑）
        for bid in np.unique(blocks_np):
            mask = (blocks_np == bid)
            block_samples = self.cma_blocks[bid].sample()
            block_fitness = fitness_values[mask]
            
            if len(block_samples) > 0:
                samples_array = np.array(block_samples)
                if samples_array.ndim == 1:
                    samples_array = samples_array.reshape(1, -1)
                    
                self.cma_blocks[bid].update(
                    all_samples=samples_array,
                    fitness_values=block_fitness
                )
        
        # 5. 动态调整稀疏度（保留原有逻辑）
        global_best = np.max(fitness_values)
        for bid in np.unique(blocks_np):
            cma = self.cma_blocks[bid]
            if cma.best_fitness < global_best:
                cma.sparsity = min(cma.sparsity * 1.1, 0.5)
            else:
                cma.sparsity = max(cma.sparsity * 0.9, 0.05)



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
            #         # === 关键修复：清空所有块的旧样本 ===
            # for bid in self.cma_blocks.values():
            #     bid.samples = []
            # 1. 生成批量扰动样本 (形状: [num_samples, C, H, W])并更新通用扰动 分块扰动
            deltas, blocks = self.sample_perturbations()
            # print("deltas.shape : ", deltas.shape)
            # print("blocks.shape : ",blocks.shape)
            # 这个函数的大体介绍：先选块号，然后对应块生成高斯噪声扰动
            # delta = delta.squeeze(0)  # 从 [1, C, H, W] -> [C, H, W]
            # delta = delta.unsqueeze(0).to(self./device)  # 确保 delta 在 GPU 上
            # delta : 形状:[num_samples, C, H, W]; blocks是就是selected_blocks,当前的状态是numpy类型。




            # print(f"\nIteration {t}, Epsilon: {self.epsilon}, Perturbation Norm: {torch.norm(self.universal_perturbation).item()}")

            # 3. 生成对抗样本
            candidate_advs = self.original_image.unsqueeze(0) + deltas  # 3D张量
            # print("attack:adv_images", adv_images.shape)

            # 4. 调用 evaluate 函数计算损失值 
            losses = self.evaluate(candidate_advs) 
            # print("losses.shape:",losses.shape)
            best_idx = torch.argmax(losses)  # 选择损失最大的候选
            fitness_values = losses.cpu().numpy()  # 适应度值（越大越好），这里是将PyTorch张量转换为NumPy数组
            # 2.更新全局扰动
            self.universal_perturbation = torch.clamp(
                self.universal_perturbation + deltas[best_idx],
                -self.epsilon, 
                self.epsilon
            )
                    # === 关键修复点：定义当前最佳对抗样本 ===
            best_adv = candidate_advs[best_idx]  # [C, H, W]
            current_adv_images = best_adv  # [C, H, W]
            if debug_mode:
                # === 新增可视化调用点1：扰动生成后 ===
                self.visualizer.plot_perturbation(
                    cid=int(self.label.item()),
                    delta=deltas[best_idx].unsqueeze(0).detach().cpu(),
                    universal_pert=self.universal_perturbation.detach().cpu(),
                    iteration=t
                )
                # ==== 新增调试代码 ====
                if epsilon == 0:
                    assert torch.allclose(self.original_image, current_adv_images, atol=1e-6), \
                        f"零扰动不一致！差异: {torch.max(torch.abs(self.original_image - current_adv_images)).item()}"
            
            #6 更新块选择概率
            # blocks,fitness_values:numpy_array 
            #传进来的应该是分过块的blocks(张量形式)和对应的fitness_values
            self.update_prob(blocks, fitness_values) #全代码中唯一一次对这个函数的调用
            

            if debug_mode:
                # === 新增可视化调用点2,3：概率更新后 ===
                self.visualizer.log_block_heatmap(
                    cid=int(self.label.item()), #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                    iteration=t,
                    selected_blocks=blocks.cpu().numpy().astype(np.int64)
                    )
                self.visualizer.log_block_distribution(
                    class_id=int(self.label.item()), #  1. 非目标攻击时，记录被攻击的原始类别 2. 目标攻击时，需改为目标类别.注：当 label 是单个数值张量时（如分类攻击的目标类别），这是正确的方法。在批量攻击时会出错！
                    iteration=t,
                    prob_dist=self.prob.float().cpu()
                )
            
                # === 新增可视化调用点4：CMA-ES更新 ===
                for bid in blocks.unique().cpu().numpy():
                    self.visualizer.visualize_cma_es(
                        cid=self.label.item(),
                        bid=bid,
                        cma_state={
                            'c1_history': self.cma_blocks[bid].c1_history,
                            'cmu_history': self.cma_blocks[bid].cmu_history,
                            'p_sigma_history': self.cma_blocks[bid].p_sigma_history,
                            'p_c_history': self.cma_blocks[bid].p_c_history,
                            'C_history': self.cma_blocks[bid].C_history,
                            'sigma_history': self.cma_blocks[bid].sigma_history
                        },
                        iteration=t
                    )
            # 7. 计算当前最佳损失值
            # === 性能监控 ===
            with torch.no_grad():
                current_loss = losses[best_idx]  # 直接使用已计算结果
                
            # 更新最佳结果（删除重复计算）
            if current_loss > self.best_loss:
                self.best_loss = current_loss
                self.best_adv = current_adv_images.detach().clone()
            
            # 9. 记录全局准确率
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
                    Loss=f"{self.best_loss:.4f}"
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

    # def visualize_cma_es(self, cid, bid, cma_state, iteration):
    #     """仅显示CMA-ES参数变化曲线"""
    #     plt.figure(figsize=(12, 6))
        
    #     # ================== 参数演化曲线 ==================
    #     # 创建2x2子图布局
    #     ax1 = plt.subplot(2, 2, 1)  # 学习率参数
    #     ax2 = plt.subplot(2, 2, 2)  # 进化路径范数
    #     ax3 = plt.subplot(2, 2, 3)  # 步长历史
    #     ax4 = plt.subplot(2, 2, 4)  # 协方差条件数
        
    #     # 确保所有历史记录存在
    #     param_history = {
    #         'c1': cma_state.get('c1_history', []),
    #         'cmu': cma_state.get('cmu_history', []),
    #         'p_sigma': cma_state.get('p_sigma_history', []),
    #         'p_c': cma_state.get('p_c_history', []),
    #         'C': cma_state.get('C_history', [])
    #     }
        
    #     # 曲线1: 学习率参数
    #     ax1.plot(param_history['c1'], color='#1f77b4', label='c1')
    #     ax1.plot(param_history['cmu'], color='#ff7f0e', label='cmu')
    #     ax1.set_title("Learning Rates")
    #     ax1.set_ylabel("Value")
    #     ax1.set_ylim(0, 1.2)
    #     ax1.legend()
        
    #     # 曲线2: 进化路径范数
    #     p_sigma_norms = [np.linalg.norm(p) for p in param_history['p_sigma']]
    #     p_c_norms = [np.linalg.norm(p) for p in param_history['p_c']]
    #     ax2.semilogy(p_sigma_norms, color='#2ca02c', label='||p_σ||')
    #     ax2.semilogy(p_c_norms, color='#d62728', label='||p_c||')
    #     ax2.set_title("Evolution Path Norms")
    #     ax2.set_ylabel("Norm (log scale)")
    #     ax2.legend()
        
    #     # 曲线3: 步长历史
    #     ax3.semilogy(cma_state['sigma_history'], color='#9467bd')
    #     ax3.set_title("Step Size History")
    #     ax3.set_xlabel("Iteration")
    #     ax3.set_ylabel("σ (log scale)")
        
    #     # 曲线4: 协方差条件数
    #     cond_numbers = [np.linalg.cond(c) for c in param_history['C']]
    #     ax4.semilogy(cond_numbers, color='#8c564b')
    #     ax4.set_title("Covariance Condition")
    #     ax4.set_xlabel("Iteration")
    #     ax4.set_ylabel("Condition Number")
        
    #     plt.tight_layout()
    #     plt.savefig(f"attack_logs/class_{cid}/cma_es/block_{bid}.png", 
    #             dpi=150, bbox_inches='tight')
    #     plt.close()
    def visualize_cma_es(self, cid, bid, cma_state, iteration):
        """将CMA-ES参数变化曲线保存到文本文件"""
        # 指定保存路径
        save_path = f"attack_logs/class_{cid}/cma_es/block_{bid}.txt"
        
        # 确保所有历史记录存在
        param_history = {
            'c1': cma_state.get('c1_history', []),
            'cmu': cma_state.get('cmu_history', []),
            'p_sigma': cma_state.get('p_sigma_history', []),
            'p_c': cma_state.get('p_c_history', []),
            'C': cma_state.get('C_history', []),
            'sigma': cma_state.get('sigma_history', [])
        }
        # print(param_history)
        # 将历史记录写入文本文件
        with open(save_path, 'w') as f:
            f.write("Iteration, c1, cmu, ||p_sigma||, ||p_c||, sigma, condition number\n")
            for i, (c1, cmu, p_sigma, p_c, C, sigma) in enumerate(zip(
                    param_history['c1'], param_history['cmu'], 
                    [np.linalg.norm(p) for p in param_history['p_sigma']], 
                    [np.linalg.norm(p) for p in param_history['p_c']], 
                    [np.linalg.cond(c) for c in param_history['C']], 
                    param_history['sigma']
                )):
                f.write(f"{i}, {c1}, {cmu}, {p_sigma}, {p_c}, {sigma}, {C}\n")

        # print(f"CMA-ES参数历史已保存到 {save_path}")
    def plot_perturbation(self, cid, delta, universal_pert, iteration):
        """完全修复热力图维度问题的可视化函数"""
        # === 1. 输入预处理 ===
        def _ensure_3d(tensor):
            tensor = tensor.squeeze()
            if tensor.dim() == 4:
                tensor = tensor[0]
            if tensor.dim() != 3:
                raise ValueError(f"输入应为3D或4D张量，实际得到: {tensor.shape}")
            return tensor

        # === 2. 安全处理 ===
        try:
            # 处理原始输入
            delta_3d = _ensure_3d(delta)
            pert_3d = _ensure_3d(universal_pert)
            
            # === 关键修复：正确的2D热力图计算 ===
            heatmap = torch.norm(pert_3d, p=2, dim=0)  # 沿通道维计算，输出[H,W]
            
            # 转换为numpy并调整对比度
            delta_np = delta_3d.permute(1, 2, 0).cpu().numpy()
            pert_np = pert_3d.permute(1, 2, 0).cpu().numpy()
            delta_np = np.clip(delta_np * 2.5 + 0.5, 0, 1)  # 缩放并安全裁剪
            pert_np = np.clip(pert_np * 2.5 + 0.5, 0, 1)  # 更温和的缩放参数
            heatmap = heatmap.cpu().numpy()

        except Exception as e:
            print(f"⚠️ 可视化预处理失败: {str(e)}")
            return

        # === 3. 绘图 ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 当前扰动
        axs[0].imshow(delta_np)
        axs[0].set_title(f"Class {cid} Iter {iteration} Delta")
        
        # 累积扰动
        axs[1].imshow(pert_np)
        axs[1].set_title(f"Class {cid} Accumulated Pert")
        
        # 热力图（确保2D）
        if heatmap.ndim != 2:
            heatmap = heatmap.mean(axis=0)  # 应急处理：取通道均值
        im = axs[2].imshow(heatmap, cmap='inferno')
        plt.colorbar(im, ax=axs[2])
        axs[2].set_title(f"Class {cid} Pert Magnitude")

        # === 4. 保存 ===
        os.makedirs(f"attack_logs/perturbations/class_{cid}", exist_ok=True)
        plt.savefig(f"attack_logs/perturbations/class_{cid}/iter_{iteration:04d}.png", 
                bbox_inches='tight', dpi=150)
        plt.close()
        torch.cuda.empty_cache()

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
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    # 初始化设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "../tiny-imagenet-200"
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
    # class_samples = {}
    # dataset = val_loader.dataset
    # for idx in range(len(dataset)):
    #     img, lbl = dataset[idx]
    #     lbl_value = lbl
    #     if lbl_value not in class_samples:
    #         # 存储图像路径信息
    #         img_pth = dataset.imgs[idx][0]
    #         label_tensor = torch.tensor(lbl, dtype=torch.long)  # 转换为张量
    #         class_samples[lbl_value] = (img, label_tensor, img_pth)
    #     if len(class_samples) >= num_test_classes:
    #         break
    class_samples = {}
    dataset = val_loader.dataset

    for idx in range(len(dataset)):
        # 获取原始数据集和索引（兼容Subset）
        if isinstance(dataset, torch.utils.data.Subset):
            original_dataset = dataset.dataset
            original_idx = dataset.indices[idx]
        else:
            original_idx = idx
        
        # 获取样本数据（保持原有方式）
        img, lbl = dataset[idx]
        
        # 获取图像路径（兼容两种数据集类型）
        img_pth = original_dataset.imgs[original_idx][0]  # 关键修改点
        
        # 存储逻辑保持不变
        lbl_value = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        if lbl_value not in class_samples:
            label_tensor = torch.tensor(lbl, dtype=torch.long)
            class_samples[lbl_value] = (img, label_tensor, img_pth)
        
        if len(class_samples) >= num_test_classes:
            break
    
    # 执行攻击
    attack_success = 0
    processed_samples = 0
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
            device=device,
            epsilon=epsilon,  # 传递全局参数,不然会使用默认值0.1
            universal_perturbation=universal_perturbation,  # 传入共享扰动
            original_path=img_path

        )
        # print("Visualizer exists:", hasattr(attacker, 'visualizer'))  # 应输出True,打印的时候这个可以放开，终端可以看到准确率和loss变化过程。
        # print("Logger exists:", hasattr(attacker, 'logger'))          # 应输出True
        adv_img = attacker.attack(val_loader, device) #这里面有update_prob()->update()：cma-es参数更新。
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
            processed_samples += 1  # 正确计数
        # 计算扰动指标
        delta = (adv_img - image).abs().sum(0)
        total_l0 += (delta > 0.005).sum().item()
        total_l1 += delta.sum().item()
        
        # 更新进度
        current_sr = (attack_success /processed_samples) * 100
        test_progress.set_postfix(Success=f"{current_sr:.1f}%")
    
    # 打印结果
    print(f"\n最终攻击成功率: {(attack_success /processed_samples)*100:.1f}%")
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