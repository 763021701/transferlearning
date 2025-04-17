"""
特征函数可视化工具

这个模块提供函数来计算和可视化特征函数，包括幅度和相位的复平面图。
可以直接从Jupyter notebook导入和使用这些函数。
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加DeepDG代码目录到路径
sys.path.append(os.path.abspath('../../'))

# 从CFD.py导入特征函数计算的函数
try:
    from alg.algs.CFD import calculate_norm, calculate_imag, calculate_real
    from alg.algs.CFD import SampleNet, net_dim_dict
except ImportError as e:
    print(f"导入CFD模块时出错: {e}")
    print("定义替代函数...")
    
    # 如果无法导入，提供替代函数
    def calculate_norm(x_r, x_i):
        return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))

    def calculate_imag(x):
        """计算特征函数的虚部"""
        # 确保我们在合适的维度上取均值（沿特征维度）
        if x.dim() <= 1:
            return torch.sin(x)
        return torch.mean(torch.sin(x), dim=1)

    def calculate_real(x):
        """计算特征函数的实部"""
        # 确保我们在合适的维度上取均值（沿特征维度）
        if x.dim() <= 1:
            return torch.cos(x)
        return torch.mean(torch.cos(x), dim=1)
    
    net_dim_dict = {"resnet18": 512, "resnet50": 2048, "vgg16": 1000}


def generate_t_values(feature_dim, num_angles=360, radius=1.0):
    """生成不同方向的t值，用于特征函数计算"""
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    t_values = []
    
    for angle in angles:
        # 创建指向特定角度的向量
        direction = np.zeros(feature_dim)
        # 为前两个维度赋值以创建方向向量，增加更多维度的变化
        direction[0] = radius * np.cos(angle)
        direction[1] = radius * np.sin(angle)
        # 添加更多维度的变化以产生更多的特征差异
        if feature_dim > 2:
            direction[2] = radius * 0.5 * np.sin(2*angle)  # 添加二次谐波
        if feature_dim > 3:
            direction[3] = radius * 0.3 * np.cos(3*angle)  # 添加三次谐波
        t_values.append(direction)
    
    return torch.Tensor(np.array(t_values))


def calculate_characteristic_function(features, t_values):
    """计算给定特征和t值的特征函数"""
    # 执行矩阵乘法 t * features.T
    inner_product = torch.matmul(t_values, features.t())
    
    # 计算实部和虚部
    t_x_real = calculate_real(inner_product)
    t_x_imag = calculate_imag(inner_product)
    
    # 计算幅度
    t_x_norm = calculate_norm(t_x_real, t_x_imag)
    
    # 打印调试信息
    print(f"Debug - inner_product shape: {inner_product.shape}")
    print(f"Debug - t_x_real shape: {t_x_real.shape}")
    print(f"Debug - t_x_imag shape: {t_x_imag.shape}")
    print(f"Debug - t_x_norm shape: {t_x_norm.shape}")
    
    # 确保返回的张量都是二维的
    if t_x_real.dim() == 1:
        t_x_real = t_x_real.unsqueeze(1)
    if t_x_imag.dim() == 1:
        t_x_imag = t_x_imag.unsqueeze(1)
    if t_x_norm.dim() == 1:
        t_x_norm = t_x_norm.unsqueeze(1)
    
    return t_x_real, t_x_imag, t_x_norm


def visualize_characteristic_function(features, domains=None, num_angles=360, radius=1.0, 
                                     domain_names=None, domain_colors=None, overlap=False):
    """在复平面上可视化特征函数
    
    Args:
        features: 特征张量 [N x D]
        domains: 域标签张量 [N]
        num_angles: 角度数量
        radius: t值的半径
        domain_names: 可选，域名称的列表，例如['Source', 'Target']
        domain_colors: 可选，域颜色的列表，例如['red', 'blue']
        overlap: 是否叠加显示不同域的分布，如为False则并排显示
    """
    feature_dim = features.shape[1]
    
    # 生成不同方向的t值
    t_values = generate_t_values(feature_dim, num_angles, radius)
    t_values = t_values.to(features.device)
    
    # 计算特征函数值
    real_values, imag_values, amplitude = calculate_characteristic_function(features, t_values)
    
    # 创建极坐标图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 转换为numpy数组以便绘图
    real_values = real_values.cpu().numpy()
    imag_values = imag_values.cpu().numpy()
    amplitude = amplitude.cpu().numpy()
    
    # 检查维度并处理
    print(f"Debug - amplitude shape: {amplitude.shape}")
    print(f"Debug - real_values shape: {real_values.shape}")
    print(f"Debug - imag_values shape: {imag_values.shape}")
    
    # 确保amplitude是二维数组
    if len(amplitude.shape) == 1:
        # 如果是一维，将其扩展为二维
        amplitude = amplitude.reshape(-1, 1)
        real_values = real_values.reshape(-1, 1)
        imag_values = imag_values.reshape(-1, 1)
        print(f"Debug - reshaped amplitude: {amplitude.shape}")
    
    # 缩放振幅以增强可视化效果
    # 找出振幅的平均值和标准差
    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    print(f"Debug - amplitude mean: {mean_amp}, std: {std_amp}")
    
    # 如果标准差很小，放大振幅变化
    if std_amp < 0.1 and mean_amp > 0.5:
        print("放大振幅变化以增强视觉效果")
        # 从0.5开始，放大变化范围
        amplitude = 0.5 + (amplitude - mean_amp) * 5  # 放大变化
    
    # 从t向量获取角度
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # 设置扇形柱状图的宽度
    width = 2 * np.pi / num_angles * 0.9  # 稍微小于角度间隔，留出间隙
    
    # 如果有域信息，用不同的颜色绘制每个域
    if domains is not None:
        unique_domains = torch.unique(domains)
        num_domains = len(unique_domains)
        
        # 如果提供了域名称，确保数量匹配
        if domain_names is not None:
            if len(domain_names) < num_domains:
                domain_names = domain_names + [f"Domain {i}" for i in range(len(domain_names), num_domains)]
            domain_names = domain_names[:num_domains]
        else:
            domain_names = [f"Domain {d.item()}" for d in unique_domains]
        
        # 如果提供了域颜色，确保数量匹配
        if domain_colors is not None:
            if len(domain_colors) < num_domains:
                # 使用默认颜色表补充
                default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
                domain_colors = domain_colors + [default_colors[i % 10] for i in range(len(domain_colors), num_domains)]
            colors = domain_colors[:num_domains]
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, num_domains))
        
        # 如果域数量为2，默认使用红色和蓝色
        if num_domains == 2 and domain_colors is None:
            colors = ['red', 'blue']
        
        # 计算叠加或并排显示的参数
        if not overlap and num_domains > 1:
            # 计算偏移量，使不同域的柱子错开
            bar_width = width / num_domains
            offsets = np.linspace(-(width/2) + (bar_width/2), (width/2) - (bar_width/2), num_domains)
        else:
            bar_width = width
            offsets = [0] * num_domains
        
        # 为确保叠加模式下所有域都可见，我们按反向顺序绘制
        # 这样第一个域（通常是Source）会显示在最上面
        if overlap:
            iter_domains = reversed(list(enumerate(unique_domains)))
        else:
            iter_domains = enumerate(unique_domains)
        
        # 保存每个域的平均振幅用于打印统计信息
        domain_amplitude_stats = {}
        
        # 将域索引映射到实际数据位置的字典
        domain_to_indices = {}
        
        # 首先收集每个域对应的样本索引
        for i, domain_id in enumerate(unique_domains):
            mask = domains == domain_id
            domain_indices = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            
            # 检查域下所有样本的索引是否都在合理范围内
            domain_to_indices[i] = domain_indices
            print(f"Domain {i} ({domain_names[i]}) has {len(domain_indices)} samples, indices range: {domain_indices.min()} to {domain_indices.max()}")
        
        # 为每个域创建柱状图
        for i, domain_id in iter_domains:
            domain_indices = domain_to_indices[i]
            
            # 验证索引的有效性
            valid_samples = []
            for sample_idx in domain_indices:
                if sample_idx < amplitude.shape[1]:
                    valid_samples.append(sample_idx)
            
            if len(valid_samples) > 0:
                print(f"Domain {domain_names[i]} ({i}): 有 {len(valid_samples)} 个有效样本")
                
                # 计算这个域的平均振幅，对所有有效样本取平均
                domain_amplitude = np.mean(amplitude[:, valid_samples], axis=1)
                
                # 保存域的统计信息
                domain_amplitude_stats[domain_names[i]] = {
                    'mean': np.mean(domain_amplitude),
                    'std': np.std(domain_amplitude),
                    'min': np.min(domain_amplitude),
                    'max': np.max(domain_amplitude)
                }
                
                # 叠加模式下使用不同的透明度设置
                if overlap:
                    # 使用半透明显示所有数据
                    transparency = 0.6
                else:
                    transparency = 0.8
                
                # 绘制扇形柱状图
                print(f"正在绘制域 {domain_names[i]} 的柱状图，offset = {offsets[i]}")
                ax.bar(angles + offsets[i], domain_amplitude, width=bar_width, 
                       alpha=transparency,
                       color=colors[i], label=domain_names[i],
                       bottom=0.0)  # 从0开始绘制
            else:
                print(f"警告: 域 {domain_names[i]} ({i}) 没有有效样本!")
        
        # 打印每个域的振幅统计信息
        print("\n域振幅统计信息:")
        for domain, stats in domain_amplitude_stats.items():
            print(f"{domain}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}")
    else:
        # 绘制平均振幅的扇形柱状图
        mean_amplitude = amplitude.mean(axis=1)
        ax.bar(angles, mean_amplitude, width=width, alpha=0.7, color='blue', bottom=0.0)
    
    # 在复平面中添加散点图以显示实部和虚部
    axc = fig.add_axes([0.7, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
    
    # 计算复平面点的边界，以便更好地显示
    complex_values = real_values + 1j * imag_values
    # 找到中心点
    mean_complex = np.mean(complex_values)
    real_center = np.real(mean_complex)
    imag_center = np.imag(mean_complex)
    
    # 找到最大偏移
    max_offset = np.max(np.abs(complex_values - mean_complex)) * 1.5
    
    # 定义step用于采样散点图的点
    step = max(1, num_angles // 36)
    
    # 如果有域信息，用不同的颜色绘制复平面中的点
    if domains is not None:
        for i, domain_id in enumerate(unique_domains):
            # 使用之前收集的域索引，避免重复计算
            domain_indices = domain_to_indices[i]
            
            if len(domain_indices) > 0:
                # 选择第一个有效样本
                valid_indices = [idx for idx in domain_indices if idx < amplitude.shape[1]]
                if valid_indices:
                    idx = valid_indices[0]  # 只选择第一个样本进行展示
                    pts = axc.scatter(real_values[::step, idx], imag_values[::step, idx], 
                                   s=15, alpha=0.8, color=colors[i], label=domain_names[i])
        
        # 为复平面图添加图例
        axc.legend(loc='best', fontsize='small')
    else:
        # 只绘制点的子集
        max_samples = min(10, amplitude.shape[1])  # 限制为10个样本或可用样本数（取较小值）
        for i in range(max_samples):
            pts = axc.scatter(real_values[::step, i], imag_values[::step, i], s=10, alpha=0.5, 
                            cmap='cool', c=angles[::step] / (2*np.pi))
    
    # 添加均值
    mean_real = real_values.mean(axis=1)
    mean_imag = imag_values.mean(axis=1)
    axc.scatter(mean_real[::step], mean_imag[::step], color='black', s=5, alpha=0.5)
    
    # 设置复平面图的范围
    axc.set_xlim(real_center - max_offset, real_center + max_offset)
    axc.set_ylim(imag_center - max_offset, imag_center + max_offset)
    
    axc.set_xlabel('Real')
    axc.set_ylabel('Imaginary')
    axc.set_title('Complex Plane')
    axc.grid(True)
    
    # 添加标签和标题
    ax.set_title('Characteristic Function Visualization')
    ax.set_rticks(np.linspace(0, 1, 5))
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 设置极坐标图的角度标签
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels([f"{int(angle)}°" for angle in np.linspace(0, 330, 12, endpoint=True)])
    
    plt.tight_layout()
    return fig, real_values, imag_values, amplitude


def extract_features(model, data_loader, device='cuda'):
    """从数据中提取特征"""
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            feature = model(data)
            features.append(feature)
            labels.append(target)
            
            # 限制为较小的批次以便可视化
            if batch_idx > 2:
                break
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def generate_random_features(num_samples=100, feature_dim=512, num_domains=4, device='cuda'):
    """生成随机特征以进行演示"""
    # 确保每个域至少有一个样本
    samples_per_domain = max(1, num_samples // num_domains)
    num_samples = samples_per_domain * num_domains  # 调整总样本数
    
    # 为不同域生成不同的分布
    all_features = []
    all_domains = []
    
    for domain_id in range(num_domains):
        # 为每个域生成略有不同的特征
        domain_features = torch.randn(samples_per_domain, feature_dim)
        
        # 给不同域的特征添加不同的偏移，增加域间差异
        shift = torch.zeros(feature_dim)
        shift[0] = 1.5 * np.cos(domain_id * np.pi / 2)  # 增加偏移量
        shift[1] = 1.5 * np.sin(domain_id * np.pi / 2)  # 增加偏移量
        
        # 为每个域添加一些特殊的特征结构
        if feature_dim > 2:
            # 添加一些域特有的模式
            pattern = torch.zeros(feature_dim)
            pattern[2:min(10, feature_dim)] = torch.linspace(0.5, 0.1, min(8, feature_dim-2))
            pattern = pattern.roll(shifts=domain_id*5)
            
            # 对每个样本应用略有差异的模式
            for i in range(samples_per_domain):
                noise_scale = 0.2 + 0.1 * (i % 3)  # 样本间的变化
                domain_features[i] += shift + pattern * (1.0 + noise_scale * torch.randn(1))
        else:
            domain_features = domain_features + shift
        
        all_features.append(domain_features)
        all_domains.append(torch.ones(domain_features.shape[0], dtype=torch.long) * domain_id)
    
    # 连接所有特征和域标签
    all_features = torch.cat(all_features, dim=0)
    all_domains = torch.cat(all_domains, dim=0)
    
    print(f"Debug - generated features shape: {all_features.shape}")
    print(f"Debug - generated domains shape: {all_domains.shape}")
    
    # 将特征移动到指定设备上
    device_name = device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU 代替")
        device_name = 'cpu'
    
    all_features = all_features.to(device_name)
    all_domains = all_domains.to(device_name)
    
    return all_features, all_domains


def load_trained_model(args):
    """加载训练好的模型"""
    try:
        from alg.modelopera import load_model
        netF, netC = load_model(args)
        print("Model loaded successfully!")
        if torch.cuda.is_available():
            netF = netF.cuda()
            netC = netC.cuda()
        netF.eval()
        netC.eval()
        return netF, netC
    except Exception as e:
        print(f"Error loading model: {e}")
        # 如果无法加载实际模型，可以创建一个随机模型用于演示
        print("Creating random model for demonstration...")
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = torch.nn.Identity()  # 移除最后的全连接层，只保留特征提取器
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model, None


# 提供一个示例Args类供Jupyter使用
class Args:
    def __init__(self):
        self.net = 'resnet18'
        self.data_dir = '../../../data'
        self.dataset = 'PACS'
        self.checkpoint_path = '../trained_models/my_model.pkl'
        self.domain = 'cartoon'
        self.batch_size = 32
        self.task = 'img_dg'


def compare_distributions(features1, features2, domain_names=None, colors=None, 
                         num_angles=72, radius=2.0, overlap=True):
    """比较两个特征分布的函数
    
    Args:
        features1: 第一个特征分布 [N1 x D]
        features2: 第二个特征分布 [N2 x D]
        domain_names: 两个分布的名称，例如 ['源域', '目标域']
        colors: 两个分布的颜色，例如 ['red', 'blue']
        num_angles: 角度数量
        radius: t值的半径
        overlap: 是否在同一位置叠加显示扇形(True)，还是并排显示(False)
    """
    # 规范化特征
    features1 = torch.nn.functional.normalize(features1, dim=1)
    features2 = torch.nn.functional.normalize(features2, dim=1)
    
    # 修正特征数据维度必须与domains完全对应
    # 计算特征的数量
    num_features1 = features1.shape[0]
    num_features2 = features2.shape[0]
    print(f"特征1数量: {num_features1}, 特征2数量: {num_features2}")
    
    # 合并两个分布
    combined_features = torch.cat([features1, features2], dim=0)
    # 创建域标签：0表示第一个分布，1表示第二个分布
    domains = torch.cat([
        torch.zeros(num_features1, dtype=torch.long),
        torch.ones(num_features2, dtype=torch.long)
    ])
    
    print(f"合并后的特征形状: {combined_features.shape}, 域标签形状: {domains.shape}")
    print(f"Source域(0)样本数: {(domains == 0).sum().item()}, Target域(1)样本数: {(domains == 1).sum().item()}")
    
    # 设置默认的域名称和颜色
    if domain_names is None:
        domain_names = ['Distribution 1', 'Distribution 2']
    if colors is None:
        colors = ['red', 'blue']
    
    # 分别计算两个分布的特征函数，然后再进行可视化比较
    feature_dim = combined_features.shape[1]
    
    # 生成不同方向的t值
    t_values = generate_t_values(feature_dim, num_angles, radius)
    
    # 分别计算两个分布的特征函数
    print("计算Source域特征函数...")
    t_values1 = t_values.to(features1.device)
    real_values1, imag_values1, amplitude1 = calculate_characteristic_function(features1, t_values1)
    
    print("计算Target域特征函数...")
    t_values2 = t_values.to(features2.device)
    real_values2, imag_values2, amplitude2 = calculate_characteristic_function(features2, t_values2)
    
    # 创建极坐标图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 转换为numpy数组以便绘图
    real_values1 = real_values1.cpu().numpy()
    imag_values1 = imag_values1.cpu().numpy()
    amplitude1 = amplitude1.cpu().numpy()
    
    real_values2 = real_values2.cpu().numpy()
    imag_values2 = imag_values2.cpu().numpy()
    amplitude2 = amplitude2.cpu().numpy()
    
    # 合并两个分布的振幅以统一缩放
    if len(amplitude1.shape) == 1:
        amplitude1 = amplitude1.reshape(-1, 1)
    if len(amplitude2.shape) == 1:
        amplitude2 = amplitude2.reshape(-1, 1)
    
    # 缩放振幅以增强可视化效果
    # 找出振幅的平均值和标准差
    all_amplitudes = np.concatenate([amplitude1, amplitude2], axis=1)
    mean_amp = np.mean(all_amplitudes)
    std_amp = np.std(all_amplitudes)
    print(f"Debug - amplitude mean: {mean_amp}, std: {std_amp}")
    
    # 如果标准差很小，放大振幅变化
    if std_amp < 0.1 and mean_amp > 0.5:
        print("放大振幅变化以增强视觉效果")
        # 从0.5开始，放大变化范围
        amplitude1 = 0.5 + (amplitude1 - mean_amp) * 5  # 放大变化
        amplitude2 = 0.5 + (amplitude2 - mean_amp) * 5  # 放大变化
    
    # 从t向量获取角度
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # 设置扇形柱状图的宽度
    width = 2 * np.pi / num_angles * 0.9  # 稍微小于角度间隔，留出间隙
    
    # 计算叠加或并排显示的参数
    if not overlap:
        # 计算偏移量，使不同域的柱子错开
        bar_width = width / 2  # 两个域
        offsets = [-bar_width/2, bar_width/2]  # 对应Source和Target
    else:
        bar_width = width
        offsets = [0, 0]
    
    # 统计两个域的振幅信息
    domain_amplitude_stats = {}
    
    # 计算和打印Source域的统计信息
    mean_amplitude1 = np.mean(amplitude1, axis=1)
    domain_amplitude_stats['Source'] = {
        'mean': np.mean(mean_amplitude1),
        'std': np.std(mean_amplitude1),
        'min': np.min(mean_amplitude1),
        'max': np.max(mean_amplitude1)
    }
    
    # 计算和打印Target域的统计信息
    mean_amplitude2 = np.mean(amplitude2, axis=1)
    domain_amplitude_stats['Target'] = {
        'mean': np.mean(mean_amplitude2),
        'std': np.std(mean_amplitude2),
        'min': np.min(mean_amplitude2),
        'max': np.max(mean_amplitude2)
    }
    
    # 设置透明度
    transparency = 0.6 if overlap else 0.8
    
    # 绘制Source域
    colors = colors or ['red', 'blue']
    labels = domain_names or ['Source', 'Target']
    
    # 在overlap模式下，先画Target再画Source，这样Source会显示在上面
    if overlap:
        # 先画Target
        ax.bar(angles + offsets[1], mean_amplitude2, width=bar_width, 
               alpha=transparency, color=colors[1], label=labels[1],
               bottom=0.0)
        # 再画Source
        ax.bar(angles + offsets[0], mean_amplitude1, width=bar_width, 
               alpha=transparency, color=colors[0], label=labels[0],
               bottom=0.0)
    else:
        # 并排模式，顺序无关紧要
        ax.bar(angles + offsets[0], mean_amplitude1, width=bar_width, 
               alpha=transparency, color=colors[0], label=labels[0],
               bottom=0.0)
        ax.bar(angles + offsets[1], mean_amplitude2, width=bar_width, 
               alpha=transparency, color=colors[1], label=labels[1],
               bottom=0.0)
    
    # 打印每个域的振幅统计信息
    print("\n域振幅统计信息:")
    for domain, stats in domain_amplitude_stats.items():
        print(f"{domain}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}")
    
    # 在复平面中添加散点图以显示实部和虚部
    axc = fig.add_axes([0.7, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
    
    # 定义step用于采样散点图的点
    step = max(1, num_angles // 36)
    
    # 计算复平面点的边界，以便更好地显示
    real_values = np.concatenate([real_values1, real_values2], axis=1)
    imag_values = np.concatenate([imag_values1, imag_values2], axis=1)
    complex_values = real_values + 1j * imag_values
    
    # 找到中心点
    mean_complex = np.mean(complex_values)
    real_center = np.real(mean_complex)
    imag_center = np.imag(mean_complex)
    
    # 找到最大偏移
    max_offset = np.max(np.abs(complex_values - mean_complex)) * 1.5
    
    # 绘制Source域的点
    if real_values1.shape[1] > 0:
        pts1 = axc.scatter(real_values1[::step, 0], imag_values1[::step, 0], 
                          s=15, alpha=0.8, color=colors[0], label=labels[0])
    
    # 绘制Target域的点
    if real_values2.shape[1] > 0:
        pts2 = axc.scatter(real_values2[::step, 0], imag_values2[::step, 0], 
                          s=15, alpha=0.8, color=colors[1], label=labels[1])
    
    # 为复平面图添加图例
    axc.legend(loc='best', fontsize='small')
    
    # 设置复平面图的范围
    axc.set_xlim(real_center - max_offset, real_center + max_offset)
    axc.set_ylim(imag_center - max_offset, imag_center + max_offset)
    
    axc.set_xlabel('Real')
    axc.set_ylabel('Imaginary')
    axc.set_title('Complex Plane')
    axc.grid(True)
    
    # 添加标签和标题
    ax.set_title('Characteristic Function Visualization')
    ax.set_rticks(np.linspace(0, 1, 5))
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 设置极坐标图的角度标签
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels([f"{int(angle)}°" for angle in np.linspace(0, 330, 12, endpoint=True)])
    
    plt.tight_layout()
    return fig, real_values, imag_values, all_amplitudes


# 示例代码
if __name__ == "__main__":
    try:
        print("开始生成随机特征...")
        # 生成两个不同分布的随机特征进行演示
        
        # 第一个分布 - Source域
        features1, _ = generate_random_features(
            num_samples=30,   # 样本数量
            feature_dim=32,   # 特征维度
            num_domains=1,    # 只有一个域
            device='cpu'      # 使用CPU以兼容性更好
        )
        print(f"生成第一个分布特征成功，形状为: {features1.shape}")
        
        # 第二个分布 - Target域，与Source有明显但适度的差异
        features2, _ = generate_random_features(
            num_samples=30,   # 样本数量
            feature_dim=32,   # 特征维度
            num_domains=1,    # 只有一个域
            device='cpu'      # 使用CPU以兼容性更好
        )
        
        # 添加合适的特征偏移使两个分布有适度差异但不至于一个完全掩盖另一个
        # 我们给第一个和第二个特征添加偏移，但不是太大的偏移
        offset = torch.zeros(features2.shape[1], dtype=features2.dtype)
        # 控制偏移量，使差异明显但不过大
        offset[0] = 0.8  # X轴偏移
        offset[1] = 0.8  # Y轴偏移
        
        # 对第二个分布稍微旋转并缩放一下特征
        # 这样会产生角度上的不同，更容易在极坐标图中区分
        for i in range(features2.shape[0]):
            # 对每个样本施加不同的变换 - 确保使用与features2相同的数据类型
            rotation = torch.tensor([
                [np.cos(np.pi/6), -np.sin(np.pi/6)],  # 旋转30度
                [np.sin(np.pi/6), np.cos(np.pi/6)]
            ], dtype=features2.dtype)  # 显式指定与features2相同的数据类型
            
            # 只变换前两个维度
            features2[i, :2] = torch.matmul(rotation, features2[i, :2])
            
        # 应用偏移
        features2 = features2 + offset
        print(f"生成第二个分布特征成功，形状为: {features2.shape}")
        
        # 可视化两个分布的比较
        print("开始可视化两个分布的比较...")
        num_angles = 72  # 每5度一个扇形
        
        # 使用叠加模式，颜色设置为更鲜明对比的颜色
        fig_overlap, _, _, _ = compare_distributions(
            features1, features2,
            domain_names=['Source', 'Target'],
            colors=['red', 'blue'],  # 红色和蓝色对比鲜明
            num_angles=num_angles,
            overlap=True  # 叠加显示
        )
        
        # 保存叠加模式的图像
        output_file_overlap = 'characteristic_function_overlap.png'
        fig_overlap.savefig(output_file_overlap, dpi=300, bbox_inches='tight')
        print(f"叠加模式可视化成功，图像已保存为 '{output_file_overlap}'")
        
        # 使用并排模式
        fig_side, _, _, _ = compare_distributions(
            features1, features2,
            domain_names=['Source', 'Target'],
            colors=['red', 'blue'],
            num_angles=num_angles,
            overlap=False  # 并排显示
        )
        
        # 保存并排模式的图像
        output_file_side = 'characteristic_function_side_by_side.png'
        fig_side.savefig(output_file_side, dpi=300, bbox_inches='tight')
        print(f"并排模式可视化成功，图像已保存为 '{output_file_side}'")
        
        # 显示图像
        plt.figure(fig_overlap.number)
        plt.show()
        
    except Exception as e:
        import traceback
        print(f"执行中出错: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        plt.close('all')
        print("完成。") 