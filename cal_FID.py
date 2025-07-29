import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import List, Union, Optional
from pytorch_fid.inception import InceptionV3

class FIDEvaluator:
    """
    基于pytorch-fid的FID评估器，支持完整梯度反向传播
    核心改进：兼容旧版本PyTorch，使用SVD实现矩阵平方根，避免依赖torch.linalg.sqrtm
    """
    
    def __init__(self, 
                 benchmark_dir: str,
                 benchmark_stats_path: str = "benchmark_fid_stats.npz",
                 image_size: int = 299,
                 device: Optional[Union[str, torch.device]] = None):

        self.benchmark_dir = benchmark_dir
        self.benchmark_stats_path = benchmark_stats_path
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
   
        self.inception_model = InceptionV3(output_blocks=[3], resize_input=False).to(self.device)
        self.inception_model.eval()
        

        self.benchmark_mu = None
        self.benchmark_cov = None
        

        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0))
        ])
        

        self.precompute_benchmark_stats()
    
    def precompute_benchmark_stats(self, force_recompute: bool = False) -> None:

        if not force_recompute and os.path.exists(self.benchmark_stats_path):
            print(f"加载已保存的基准统计量: {self.benchmark_stats_path}")
            stats = np.load(self.benchmark_stats_path)
            self.benchmark_mu = torch.from_numpy(stats["mu"]).float().to(self.device)
            self.benchmark_cov = torch.from_numpy(stats["cov"]).float().to(self.device)
            return
        
        print(f"预计算基准数据集统计量...")
        
        # 加载基准数据集
        dataset = ImageFolderDataset(self.benchmark_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        # 提取特征
        all_features = []
        with torch.no_grad():
            for imgs in dataloader:
                imgs = imgs.to(self.device, dtype=torch.float32) / 255.0  # [0,255] -> [0,1]
                features = self.inception_model(imgs)[0].squeeze(-1).squeeze(-1)  # [B, 2048]
                all_features.append(features)
        
        # 计算并保存统计量
        all_features = torch.cat(all_features, dim=0)
        self.benchmark_mu = torch.mean(all_features, dim=0)
        self.benchmark_cov = torch.cov(all_features.T)
        np.savez(
            self.benchmark_stats_path,
            mu=self.benchmark_mu.cpu().numpy(),
            cov=self.benchmark_cov.cpu().numpy()
        )
        
        print(f"基准统计量已保存至: {self.benchmark_stats_path}")
    
    def compute_fid(self, generated_images: torch.Tensor) -> torch.Tensor:

        if self.benchmark_mu is None or self.benchmark_cov is None:
            raise RuntimeError("请先调用precompute_benchmark_stats()")
        

        processed_imgs = self._preprocess_generated_images(generated_images)
        

        with torch.set_grad_enabled(True):

            for param in self.inception_model.parameters():
                param.requires_grad_(False)
            

            features = self.inception_model(processed_imgs)[0].squeeze(-1).squeeze(-1)
        

        gen_mu = torch.mean(features, dim=0)
        gen_cov = torch.cov(features.T)
        

        mean_diff_squared = torch.sum((self.benchmark_mu - gen_mu) **2)
        

        eps = 1e-6
        cov_benchmark = self.benchmark_cov + eps * torch.eye(self.benchmark_cov.shape[0], device=self.device)
        cov_gen = gen_cov + eps * torch.eye(gen_cov.shape[0], device=self.device)
        

        cov_product = torch.matmul(cov_benchmark, cov_gen)
        

        covmean = self._matrix_sqrtm(cov_product, eps=eps)
        

        if covmean.is_complex():
            covmean = covmean.real
        
        trace_term = torch.trace(cov_benchmark + cov_gen - 2 * covmean)
        

        fid_score = mean_diff_squared + trace_term
        return fid_score
    
    def _matrix_sqrtm(self, mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

        # 对矩阵进行SVD分解：mat = U * S * V^T
        u, s, vh = torch.svd(mat)
        
        # 数值稳定性处理：确保奇异值非负（避免sqrt(-0)）
        s = torch.clamp(s, min=eps)
        
        # 计算奇异值的平方根
        s_sqrt = torch.sqrt(s)
        
        # 重构矩阵平方根：sqrt(mat) = U * sqrt(S) * V^T
        sqrt_mat = torch.matmul(torch.matmul(u, torch.diag(s_sqrt)), vh)
        
        return sqrt_mat
    
    def _preprocess_generated_images(self, images: torch.Tensor) -> torch.Tensor:


        images = images.to(dtype=torch.float32, device=self.device)
        if not images.requires_grad:
            images = images.requires_grad_(True)
        
        # 范围转换：[-1,1] -> [0,1] 或保持[0,1]不变
        if images.min() < 0:
            images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # 尺寸调整（使用可微分的插值）
        if images.shape[2] != self.image_size or images.shape[3] != self.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        return images  # 输出形状：[N, 3, 299, 299]，范围[0,1]


class ImageFolderDataset(Dataset):
    """基准数据集加载器，返回uint8张量（0-255）"""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_paths = self._collect_image_paths()
        self.resize = transforms.Resize((299, 299), transforms.InterpolationMode.BILINEAR)
        
    def _collect_image_paths(self) -> List[str]:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        if not image_paths:
            raise ValueError(f"在{self.root_dir}中未找到任何图像文件")
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.resize(img)  # 调整为299x299
        img_np = np.array(img, dtype=np.uint8)  # [H, W, 3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # [3, H, W]
        return img_tensor
