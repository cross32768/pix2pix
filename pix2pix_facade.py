#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

print('PyTorch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
use_gpu = torch.cuda.is_available()
print('Is GPU available:', use_gpu)


# In[4]:


# 各種設定

# デバイス
device = torch.device('cuda' if use_gpu else 'cpu')
print(device)

# バッチサイズ
batchsize = 1

# シード値の設定
seed = 1
torch.manual_seed(seed)
if use_gpu:
    torch.cuda.manual_seed(seed)
    
# 学習データが置いてあるディレクトリ
data_dir = '../../data/facades/'
train_data_dir = data_dir + 'train/'
test_data_dir  = data_dir + 'test/'
val_data_dir   = data_dir + 'val/'

# 生成画像を置くディレクトリ
output_dir = data_dir + 'generated'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
# 定期的に保存する重みやオプティマイザのstate_dictなんかを置くディレクトリ
save_dir = './save/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# In[5]:


# ディレクトリ名を指定してデータを読み込み、TensorDatasetを作成する関数
# label_to_facadeがTrueならラベル→画像、Falseなら画像→ラベルを学習する
def make_TensorDataset(dir_name, label_to_facade = True):
    data_names = os.listdir(dir_name)
    Xs = []
    Ys = []
    for data_name in data_names:
        image_np = np.array(Image.open(dir_name + data_name))
        # 訓練データは左側が画像、右側がラベルになっている(サイズは厳密に同じ)
        if label_to_facade:
            X = image_np[:,  image_np.shape[1] // 2:, :]
            Y = image_np[:, :image_np.shape[1] // 2 , :]
        else:
            X = image_np[:, :image_np.shape[1] // 2 , :]
            Y = image_np[:,  image_np.shape[1] // 2:, :]
        
        Xs.append(X)
        Ys.append(Y)
    
    # 画像の輝度値は元々0〜255になっているので、-1〜1に正規化しておく
    Xs = (np.array(Xs) - 127.5) / 127.5 
    Ys = (np.array(Ys) - 127.5) / 127.5
    
    # PyTorchの[channel, height, width]の形式に合うように次元を入れ替えておく
    Xs = np.transpose(Xs, [0, 3, 1, 2])
    Ys = np.transpose(Ys, [0, 3, 1, 2])

    # TensorDatasetを作成
    dataset = TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(Ys).float())
    
    return dataset


# In[6]:


# TensorDatasetを作成して、データ数を表示する
train_data = make_TensorDataset(train_data_dir)
val_data = make_TensorDataset(val_data_dir)

print('The number of training data:', len(train_data))
print('The number of valdation data:', len(val_data))


# In[7]:


# データローダを用意
train_loader = DataLoader(train_data, batch_size = batchsize, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batchsize, shuffle = False)


# In[10]:


'''
# 訓練データのうちの1つを表示してみる
Xs, Ys = iter(train_loader).next()

# 入力
print('The shape of input:', Xs.size()[1:])
X_example = Xs[0]*0.5 + 0.5 # -1~1を表示用に0~1に戻す
X_example = X_example.numpy()
plt.imshow(np.transpose(X_example, [1, 2, 0])) # 表示用に[height, width, channel]の形式に戻す
'''


# In[11]:


'''
# ラベル
print('The shape of label:', Ys.size()[1:])
Y_example = Ys[0]*0.5 + 0.5
Y_example = Y_example.numpy()
plt.imshow(np.transpose(Y_example, [1, 2, 0]))
'''


# In[14]:


# 便宜上のパーツを定義
# 論文のappendixでCk/CDkとして言及されているもののうち、encoderとdiscriminatorで使うダウンサンプリングの部分
# use_batchnormでバッチノルムを適用するか、use_dropoutでドロップアウト(0.5)を適用するかを指定できる
# 畳み込み→バッチノルム→ドロップアウト→leakyreluの順
# 活性化関数は0.2のleakyrelu
class CDk_downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = True, use_dropout = False):
        super(CDk_downsample, self).__init__()
        self.cv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(0.5)
        self.rl = nn.LeakyReLU(0.2)
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
    def forward(self, x):
        out = self.cv(x)
        
        if self.use_batchnorm:
            out = self.bn(out)
        
        if self.use_dropout:
            out = self.dr(out)
        
        out = self.rl(out)
        return out
    
# 論文のappendixでCk/CDkとして言及されているもののうち、decoderで使うアップサンプリングの部分
# use_batchnormでバッチノルムを適用するか、use_dropoutでドロップアウト(0.5)を適用するかを指定できる
# 畳み込み→バッチノルム→ドロップアウト→reluの順
# 活性化関数はrelu
class CDk_upsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = True, use_dropout = False):
        super(CDk_upsample, self).__init__()
        self.tc = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(0.5)
        self.rl = nn.ReLU()
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
    def forward(self, x):
        out = self.tc(x)
        
        if self.use_batchnorm:
            out = self.bn(out)
        
        if self.use_dropout:
            out = self.dr(out)
            
        out = self.rl(out)
        return out


# In[15]:


# pix2pixのGeneratorを定義
# U-net Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder_C1  = CDk_downsample(  3,  64, use_batchnorm = False)
        self.encoder_C2  = CDk_downsample( 64, 128)
        self.encoder_C3  = CDk_downsample(128, 256)
        self.encoder_C4  = CDk_downsample(256, 512)
        self.encoder_C5  = CDk_downsample(512, 512)
        self.encoder_C6  = CDk_downsample(512, 512)
        self.encoder_C7  = CDk_downsample(512, 512)
        self.encoder_C8  = CDk_downsample(512, 512, use_batchnorm = False)
        
        self.decoder_CD1 = CDk_upsample( 512    ,  512, use_batchnorm = True, use_dropout = True)
        self.decoder_CD2 = CDk_upsample(1024    , 1024, use_batchnorm = True, use_dropout = True)
        self.decoder_CD3 = CDk_upsample(1024+512, 1024, use_batchnorm = True, use_dropout = True)
        self.decoder_C4  = CDk_upsample(1024+512, 1024)
        self.decoder_C5  = CDk_upsample(1024+512, 1024)
        self.decoder_C6  = CDk_upsample(1024+256,  512)
        self.decoder_C7  = CDk_upsample( 512+128,  256)
        self.decoder_C8  = CDk_upsample( 256+ 64,  128)
        
        self.decoder_final = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
        self.th = nn.Tanh()
        
    def forward(self, x):
        out_encoder_C1  = self.encoder_C1(x)
        out_encoder_C2  = self.encoder_C2(out_encoder_C1)
        out_encoder_C3  = self.encoder_C3(out_encoder_C2)
        out_encoder_C4  = self.encoder_C4(out_encoder_C3)
        out_encoder_C5  = self.encoder_C5(out_encoder_C4)
        out_encoder_C6  = self.encoder_C6(out_encoder_C5)
        out_encoder_C7  = self.encoder_C7(out_encoder_C6)
        out_encoder_C8  = self.encoder_C8(out_encoder_C7)
        
        out_decoder_CD1 = self.decoder_CD1(out_encoder_C8)
        out_decoder_CD2 = self.decoder_CD2(torch.cat([out_decoder_CD1, out_encoder_C7], dim=1))
        out_decoder_CD3 = self.decoder_CD3(torch.cat([out_decoder_CD2, out_encoder_C6], dim=1))
        out_decoder_C4  = self.decoder_C4( torch.cat([out_decoder_CD3, out_encoder_C5], dim=1))
        out_decoder_C5  = self.decoder_C5( torch.cat([out_decoder_C4 , out_encoder_C4], dim=1))
        out_decoder_C6  = self.decoder_C6( torch.cat([out_decoder_C5 , out_encoder_C3], dim=1))
        out_decoder_C7  = self.decoder_C7( torch.cat([out_decoder_C6 , out_encoder_C2], dim=1))
        out_decoder_C8  = self.decoder_C8( torch.cat([out_decoder_C7 , out_encoder_C1], dim=1))
        
        out = self.decoder_final(out_decoder_C8)
        out = self.th(out)
        
        return out


# In[16]:


# pix2pixのdiscriminatorを定義
# patchGAN discriminator（パッチサイズがどうなってるのかよくわからん）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.C1_x = CDk_downsample(  3,  64, use_batchnorm = False)
        self.C1_y = CDk_downsample(  3,  64, use_batchnorm = False)
        self.C2   = CDk_downsample(128, 128)
        self.C3   = CDk_downsample(128, 256)
        self.C4   = CDk_downsample(256, 512)
        self.conv_final = nn.Conv2d(512, 1, kernel_size = 4, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        out_x = self.C1_x(x)
        out_y = self.C1_y(y)
        out = torch.cat([out_x, out_y], dim=1)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.conv_final(out)
        out = self.sigmoid(out)
        return out


# In[17]:


# ネットワークを実体化、オプティマイザを定義
generator = Generator()
discriminator = Discriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)

gen_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, betas = [0.5, 0.999])
dis_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0002, betas = [0.5, 0.999])

num_trainable_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
num_trainable_params_dis = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

print('--------------------------------------------------------------------------------------')
print('Generator')
print('The number of trainable parameters:', num_trainable_params_gen)
print('\nModel\n', generator)
print('\nOptimizer\n', gen_optimizer)

print('--------------------------------------------------------------------------------------')
print('Discriminator')
print('The number of trainable parameters:', num_trainable_params_dis)
print('\nModel\n', discriminator)
print('\nOptimizer\n', dis_optimizer)


# In[18]:


def gen_loss(gen_output, dis_output_fake, input_label):
    λ = 100
    adversarial_loss = F.binary_cross_entropy(dis_output_fake, torch.ones_like(dis_output_fake))
    l1_loss = F.l1_loss(gen_output, input_label)
    return adversarial_loss + λ*l1_loss
    
def dis_loss(dis_output_real, dis_output_fake):
    real_loss = F.binary_cross_entropy(dis_output_real, torch.ones_like(dis_output_real))
    fake_loss = F.binary_cross_entropy(dis_output_fake, torch.zeros_like(dis_output_fake))
    adversarial_loss =  0.75 * (real_loss + fake_loss)
    return adversarial_loss


# In[19]:


def train(data_loader):
    generator.train()
    discriminator.train()
    
    running_generator_loss = 0
    running_discriminator_loss = 0
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        gen_outputs = generator(inputs)
        dis_outputs_real = discriminator(labels, inputs)
        dis_outputs_fake_for_dis = discriminator(gen_outputs.detach(), inputs)
        dis_outputs_fake_for_gen = discriminator(gen_outputs, inputs)
        
        dis_optimizer.zero_grad()
        discriminator_loss = dis_loss(dis_outputs_real, dis_outputs_fake_for_dis)
        discriminator_loss.backward()
        dis_optimizer.step()
        
        gen_optimizer.zero_grad()
        generator_loss = gen_loss(gen_outputs, dis_outputs_fake_for_gen, labels)
        generator_loss.backward()
        gen_optimizer.step()
                                                
        running_generator_loss += generator_loss.item()
        running_discriminator_loss += discriminator_loss.item()
                                                
    average_generator_loss = running_generator_loss / len(data_loader)
    average_discriminator_loss = running_discriminator_loss / len(data_loader)
    
    return average_generator_loss, average_discriminator_loss


# In[21]:


def val(data_loader, epoch):
    n_save_images = 8
    generator.train()
    discriminator.train()
    
    running_generator_loss = 0
    running_discriminator_loss = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            gen_outputs = generator(inputs)
            dis_outputs_real = discriminator(labels, inputs)
            dis_outputs_fake = discriminator(gen_outputs, inputs)
       
            running_discriminator_loss += dis_loss(dis_outputs_real, dis_outputs_fake).item()
            running_generator_loss += gen_loss(gen_outputs, dis_outputs_fake, labels).item()
            
            if epoch % 2 == 0:
                if i >= len(data_loader) - n_save_images:
                    comparison = torch.cat([inputs, labels, gen_outputs])
                    save_image(comparison.data.cpu(), '{}/{}_{}.png'.format(output_dir, epoch, i))
                
    average_generator_loss = running_generator_loss / len(data_loader)
    average_discriminator_loss = running_discriminator_loss / len(data_loader)

    return average_generator_loss, average_discriminator_loss


# In[15]:


n_epochs = 20
train_loss_list = [[],[]]
val_loss_list = [[],[]]

for epoch in range(n_epochs):
    train_gen_loss, train_dis_loss = train(train_loader)
    val_gen_loss, val_dis_loss = val(val_loader, epoch)
    
    train_loss_list[0].append(train_gen_loss)
    train_loss_list[1].append(train_dis_loss)
    val_loss_list[0].append(val_gen_loss)
    val_loss_list[1].append(val_dis_loss)
    
    if epoch % 2 == 0:
        torch.save(generator.state_dict(), save_dir + 'generator_' + str(epoch) + '.pth')
        torch.save(discriminator.state_dict(), save_dir + 'discriminator_' + str(epoch) + '.pth')
        torch.save(gen_optimizer.state_dict(), save_dir + 'gen_optmizer_' + str(epoch) + '.pth')
        torch.save(dis_optimizer.state_dict(), save_dir + 'gen_discriminator' + str(epoch) + '.pth')
    
    print('epoch[%d/%d] losses[train_gen:%1.4f train_dis:%1.4f  val_gen:%1.4f val_dis:%1.4f]'                     % (epoch+1, n_epochs, train_gen_loss, train_dis_loss, val_gen_loss, val_dis_loss))
    
np.save(save_dir +  'train_loss_list.npy', np.array(train_loss_list))
np.save(save_dir + 'validation_loss_list.npy', np.array(val_loss_list))


# In[ ]:





# In[ ]:




