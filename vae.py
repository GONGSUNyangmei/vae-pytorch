import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor, Normalize
import numpy as np
import imageio
import os
from PIL import Image

class ResizeImage:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, image):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(self.new_size)
        resized_array = np.array(resized_image)
        return resized_array

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        j = 0
        for filename in os.listdir(data_path):
            for i in os.listdir(os.path.join(data_path, filename)):
                img = imageio.v2.imread(os.path.join(data_path, filename, i))
                if self.transform:
                    img = self.transform(img)
                self.data.append(img)
                self.labels.append(int(filename))
                j += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

pic_num = 2100
batch_size = 20
latent_dim = 30
epochs = 10
img_dim = 28
filters = 16
intermediate_dim = 256
lamb = 2.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = ResizeImage((img_dim, img_dim))
train_dataset = CustomDataset('/home/cluster_result/FZVariable-FengYaKaiSongJ', transform=transform)
test_dataset = CustomDataset('/home/tester', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



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

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_dim, filters):
        super(Decoder, self).__init__()
        # 计算输入到第一个 conv_transpose2d 层时的特征图大小
        feature_map_size = img_dim  # 假设转置卷积开始时的特征图尺寸为 img_dim 
        #num_features_before_conv = filters * self.feature_map_size * self.feature_map_size
             
        # 根据错误信息 "mat1 and mat2 shapes cannot be multiplied (17x10 and 28x128)"
        # 我们知道转置卷积层需要的特征大小可能是 img_dim(例如28) * 128
        # 让我们确定正确的feature_map_size和filters值
        fc_input_features = 28 * 128  # 这是假设的并需要根据你的网络计算
        self.model = nn.Sequential(
            nn.Linear(latent_dim, filters * 2 * (img_dim // 4) * (img_dim // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (filters * 2, img_dim // 4, img_dim // 4)),
            nn.ConvTranspose2d(filters * 2, filters * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(filters * 2, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(filters, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

class VAELoss(nn.Module):
    def __init__(self, lamb):
        super(VAELoss, self).__init__()
        self.lamb = lamb

    def forward(self, x, x_recon, z_mean, z_log_var):
        # 计算重构损失和 KL 散度
        recon_loss = F.binary_cross_entropy(x_recon, x,reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # 计算总损失
        loss = recon_loss + self.lamb * kl_loss
        return loss




# 将模型转移到定义的设备上（GPU或CPU）


encoder = Encoder( latent_dim,img_dim, filters)
decoder = Decoder(latent_dim, img_dim, filters)
vae = VAE(encoder, decoder,latent_dim)
vae.to(device)
vae_loss = VAELoss(lamb)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 循环训练
for epoch in range(epochs):
    vae.train()  # 将模型设置为训练模式
    train_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.float()
        if x_batch.max() > 1.0:
            x_batch /= 255.0  # 确保 x_batch 中的值是在 [0, 1] 范围内
        x_batch = x_batch.float().unsqueeze(1).to(device)  # 转移到设备上
        optimizer.zero_grad()  # 清除之前的梯度
        x_recon, z_mean, z_log_var = vae(x_batch)
        loss = vae_loss(x_batch, x_recon, z_mean, z_log_var)  # 计算损失
        loss.backward()  # 反向传播
        train_loss += loss.item()
        optimizer.step()  # 更新参数
    print(f'Epoch {epoch}, Loss: {train_loss/len(train_loader)}')

torch.save(encoder, 'encoder_model.pth')

# 测试
vae.eval()  # 将模型设置为评估模式
test_loss = 0
with torch.no_grad():  # 在测试阶段，不需要计算梯度
    for i, (x_batch, _) in enumerate(test_loader):
        x_batch = x_batch.float()
        if x_batch.max() > 1.0:
            x_batch /= 255.0  # 确保 x_batch 中的值是在 [0, 1] 范围内
        x_batch = x_batch.float().unsqueeze(1).to(device)  # 转移到设备上
        x_recon, z_mean, z_log_var= vae(x_batch)
        loss = vae_loss(x_batch, x_recon, z_mean, z_log_var)
        test_loss += loss.item()
        
        # 保存重构图片
        if i == 0:  # 例如，我们只保存第一批图片
            # 保存第一个批量的重构图像
            x_recon_images = x_recon.squeeze().cpu().numpy()
            os.makedirs('recon_images', exist_ok=True)
            for j, img in enumerate(x_recon_images):
                imageio.imwrite(f'recon_images/recon_image_{j}.png', (img * 255).astype(np.uint8))
            
print(f'Test Loss: {test_loss/len(test_loader)}')
