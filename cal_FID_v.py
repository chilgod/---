import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import List, Union, Optional


class SemanticDistanceEvaluator:
    """
    语义距离评估器：计算输入VAE潜变量与基准数据集B的语义距离，并支持标准化处理
    支持两种语义空间：InceptionV3特征空间 / VAE潜空间
    """
    def __init__(self, 
                 benchmark_dir: str,
                 vae_model,
                 semantic_space: str = "inception",  # "inception"或"vae"
                 benchmark_stats_path: str = "benchmark_stats.pt",
                 image_size: int = 299,
                 normalize_range: tuple = (0.0, 1.0),
                 epsilon: float = 0.1,
                 device: Optional[Union[str, torch.device]] = None):
        """初始化评估器"""
        self.benchmark_dir = benchmark_dir
        self.vae = vae_model.to(device)
        self.semantic_space = semantic_space
        self.benchmark_stats_path = benchmark_stats_path
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_range = normalize_range
        self.epsilon = epsilon
        

        if semantic_space == "inception":

            self.inception = models.inception_v3(pretrained=True, transform_input=False)
            self.inception.eval()
            self.inception.to(self.device)

            self.inception.fc = torch.nn.Identity()
            

            self.preprocess = transforms.Compose([
                transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
        

        self.benchmark_mean = None
        self.dist_max = None
        self.dist_min = None
        
        # 冻结参数
        self.vae.eval()
        if semantic_space == "inception":
            for param in self.inception.parameters():
                param.requires_grad = False

    def precompute_benchmark_stats(self, force_recompute: bool = False) -> None:
        """预计算基准数据集的统计量"""
        if not force_recompute and os.path.exists(self.benchmark_stats_path):
            print(f"加载已保存的基准统计量: {self.benchmark_stats_path}")
            stats = torch.load(self.benchmark_stats_path, map_location=self.device)
            self.benchmark_mean = stats["mean_vector"]
            self.dist_max = stats["dist_max"]
            self.dist_min = stats["dist_min"]
            return
        
        print(f"预计算基准数据集在{self.semantic_space}空间的统计量...")
        
        # 加载基准数据集
        dataset = ImageFolderDataset(self.benchmark_dir, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        # 收集特征和距离
        all_features = []
        all_distances = []
        
        with torch.no_grad():

            for imgs in dataloader:
                imgs = imgs.to(self.device)
                
                if self.semantic_space == "vae":
                    # VAE潜空间特征
                    z = self.vae.encode(imgs).latent_dist.mean
                    all_features.append(z)
                else:

                    imgs_preprocessed = self.preprocess(imgs)
                    feat = self.inception(imgs_preprocessed)
                    all_features.append(feat)
            
            # 计算基准均值
            all_features = torch.cat(all_features, dim=0)
            self.benchmark_mean = torch.mean(all_features, dim=0)
            

            for imgs in dataloader:
                imgs = imgs.to(self.device)
                
                if self.semantic_space == "vae":
                    z = self.vae.encode(imgs).latent_dist.mean
                    distances = torch.norm(z - self.benchmark_mean, dim=1)
                else:
                    imgs_preprocessed = self.preprocess(imgs)
                    feat = self.inception(imgs_preprocessed)
                    distances = torch.norm(feat - self.benchmark_mean, dim=1)
                
                all_distances.append(distances)
        
        # 统计距离范围
        all_distances = torch.cat(all_distances, dim=0)
        self.dist_max = torch.max(all_distances).item()
        self.dist_min = torch.min(all_distances).item()
        
        # 保存统计量
        torch.save({
            "mean_vector": self.benchmark_mean,
            "dist_max": self.dist_max,
            "dist_min": self.dist_min
        }, self.benchmark_stats_path)
        
        print(f"基准统计量已保存至: {self.benchmark_stats_path}")
        print(f"基准距离范围: [{self.dist_min:.4f}, {self.dist_max:.4f}]")

    def compute_semantic_distance(self, input_latents: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """计算输入潜变量与基准的语义距离"""
        if self.benchmark_mean is None or self.dist_max is None or self.dist_min is None:
            raise RuntimeError("wrong")
        
        #with torch.no_grad():
        if self.semantic_space == "vae":
                # VAE空间直接计算距离
            distances = torch.norm(input_latents - self.benchmark_mean, dim=1)
        else:

            decoded_images = self.vae.decode(input_latents).sample
            decoded_images = torch.clamp(decoded_images, 0.0, 1.0)
            imgs_preprocessed = self.preprocess(decoded_images)
            input_features = self.inception(imgs_preprocessed)
            distances = torch.norm(input_features - self.benchmark_mean, dim=1)
        
        # 标准化
        if normalize:
            dist_range = self.dist_max - self.dist_min
            if dist_range < 1e-6:
                dist_range = 1e-6
            
            normalized = (distances - self.dist_min) / dist_range
            normalized = normalized * (self.normalize_range[1] - self.normalize_range[0]) + self.normalize_range[0]
            
            if self.epsilon > 0:
                normalized = normalized * (1 - self.epsilon) + self.epsilon
            
            return normalized
        else:
            return distances


class ImageFolderDataset(Dataset):
    """简单的图像文件夹数据集"""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._collect_image_paths()
        
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
        
        if self.transform:
            img = self.transform(img)
            
        return img
