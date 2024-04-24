
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import shutil

class Encoder(nn.Module):
    def __init__(self, latent_dim, img_dim, filters):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, stride=2, padding=1),#[14,14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),#[14,14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1),#[7,7]
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters * 2, filters * 2, kernel_size=3, stride=1, padding=1),#[7,7]
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        #self.feature_map_size = self._determine_feature_map_size(img_dim)

        # your flatten layer will be of size (filters * 2 * feature_map_size^2)
        # make sure this matches intermediate_dim
        #intermediate_dim = filters * 2 * (self.feature_map_size ** 2)
        # self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        # self.fc_log_var = nn.Linear(intermediate_dim, latent_dim)
        self.fc_mu = nn.Linear(filters * 2 * img_dim * img_dim //4 //4, latent_dim)
        self.fc_log_var = nn.Linear(filters * 2 * img_dim * img_dim //4 //4 ,latent_dim)
    
    def _determine_feature_map_size(self, img_dim):
        # 模拟卷积层操作的结果，来确定最终特征图的尺寸
        size = img_dim
        for _ in range(2):  # number of times where stride=2 is used
            size = (size + 1) // 2
        return size
      
    def forward(self, x):
        h = self.conv_layers(x)
        h = torch.flatten(h, start_dim=1)
        z_mean = self.fc_mu(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
            transforms.Resize((28, 28)),  # 调整图片尺寸为 28x28
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #print(image_path)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.image_paths)

# 定义ImageDataset实例
dataset = ImageDataset(root_dir='/home/FZVariable-FengYaKaiSongJ')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载模型
encoder = torch.load('encoder_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()

# 提取特征和路径
features = []
paths = []
with torch.no_grad():
    for image, image_path in dataloader:
        image = image.to(device)
        z_mean, _ = encoder(image)
        features.append(z_mean.cpu().numpy())
        paths.append(image_path[0])

        # 层次聚类
features = np.vstack(features)
agg_clustering = AgglomerativeClustering(n_clusters=7)
cluster_labels = agg_clustering.fit_predict(features)

output_root = '/home/my_test_clustered'
os.makedirs(output_root, exist_ok=True)
for path, cluster_label in zip(paths, cluster_labels):
    cluster_folder = os.path.join(output_root, str(cluster_label))
    os.makedirs(cluster_folder, exist_ok=True)
    shutil.copy2(path, cluster_folder)