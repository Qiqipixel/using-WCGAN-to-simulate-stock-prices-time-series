# -*- coding: utf-8 -*-
"""
GAN生成模型：WGAN-GP

参考资料：
    参考Takahashi et al. 2019. Modeling financial time-series with generative adversarial networks
    https://github.com/stakahashy/fingan/

函数说明：
    build_gan_generator 构建WGAN-GP生成器
    build_gan_discriminator 构建WGAN-GP判别器
    train_gan 训练WGAN-GP模型
    simu_gan WGAN-GP生成序列   
    
"""

import numpy as np

import torch
import torch.nn as nn

from Gan_Simulator import set_random_seed, get_loader_cgan_2, plot_save_train, plot_monte,plot_simu


#%%构建CGAN-GP生成器

class build_cgan_generator(nn.Module):
    """
    Cgan生成器结构：
    输入层：gan_dim_latent个随机数（batch_size*gan_doim_latent）
    第i隐藏层：gan_dim_hidden_i个Tanh神经元
    输出层：gan_dim_output
    """
    def __init__(self,param): # 类初始化
        super(build_cgan_generator,self).__init__()
        kernel_size = 9
        #全连接层
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.Tanh())
            return layers
        ###直接进行连接
        self.Net = nn.Sequential(
            *block(param.gan_dim_latent+param.label_size , param.gan_dim_hidden_1),
            *block(param.gan_dim_hidden_1, param.gan_dim_hidden_2),
            nn.Linear(param.gan_dim_hidden_2, param.gan_dim_output),
            )

    def forward(self,label,x): # 前向传播+concat
        out = torch.cat((x, label), dim=1) ##concat之后，维数变为param.gan_dim_latent+ param.gan_lable_latent(100+128)
        return self.Net(out)      

#%% 构建CGAN-GP判别器
class build_cgan_discriminator(nn.Module):
    """
    WGAN-GP判别器结构：
    输入层：gan_dim_output个真实值/生成器生成值
    第i隐藏层：gan_dim_hidden_i个Tanh神经元
    输出层：1个神经元，删除sigmoid处理
    """
    def __init__(self,param): # 类初始化
        super(build_cgan_discriminator,self).__init__()
        kernel_size = 9
        def block(in_chan, out_chan, normalize=True):
            #1维卷积层+LeakyReLU激活函数
            layers = [nn.Conv1d(in_chan, out_chan, kernel_size=kernel_size, padding=(kernel_size-1)//2)]
            if param.batch_norm:
                layers.append(nn.BatchNorm1d(out_chan))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        self.Net = nn.Sequential(
            *block(1, 64, normalize=False),
            *block(64, 128, normalize=False),
            *block(128, 128, normalize=False),
            nn.Flatten(),
            nn.Linear(128*(param.gan_dim_output+param.label_size ), 32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32,1)
            )
    def forward(self,label,x): # 前向传播+concat
        out = torch.cat((x, label.reshape((label.size()[0],1,label.size()[1]))), dim=2) ##concat之后，维数变为param.gan_dim_latent+ param.gan_lable_latent(100+128)
        return self.Net(out)
#%%计算梯度下降
def compute_gradient_penalty(para,lebel,netD, real_data, fake_data):
        batch_size = para.batch_size
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1).expand_as(real_data).to(real_data.device)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = netD(lebel,interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

#%% 训练GAN
def train_gan(param,train_data,mute,plot):
    """
    训练GAN，返回生成器G和训练损失结果res

    Parameters
    ----------
    param : Class(Param)
        参数类.
    train_data : T*1 ndarray
        收益率或价格序列.

    Returns
    -------
    G : torch.nn.Module的子类
        生成器.
    res : num_epochs*2 DataFrame
        d_loss和g_loss.

    """
    # 0. 设置随机数种子点
    set_random_seed(param.seed)
    
    # 1. 定义数据生成器
    train_loader = get_loader_cgan_2(param,
                                   train_data,
                                   window_width_real=param.gan_dim_output,
                                   window_width_lable=param.label_size,
                                   batch_size=param.batch_size,
                                   shuffle=True)
    
    # 2. 定义网络结构
    D = build_cgan_discriminator(param).to(param.device)
    G = build_cgan_generator(param).to(param.device)
    
    # 3. 定义损失函数和优化器
    #loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-5,betas=(0.1,0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4,betas=(0.1,0.999))
    
    # 4. 训练
    # 初始化结果
    res = {'d_loss':np.zeros((param.num_epochs)),
           'g_loss':np.zeros((param.num_epochs))}
    # 训练模式
    D.train()
    G.train()
    # 逐轮迭代
    for epoch in range(param.num_epochs):
        for _ in range(param.critic_iterations):
            # a. 训练判别器
            # a1. 生成真样本
            real_data = next(iter(train_loader))[1].to(param.device)
            # 判别器含卷积操作，需要三维数据，故真样本二维转三维
            real_data_label=next(iter(train_loader))[0].to(param.device)##真的标签
            real_data = torch.reshape(real_data,(real_data.shape[0],1,real_data.shape[1]))
            #real_data_label=torch.reshape(real_data_label,(real_data_label.shape[0],1,real_data_label.shape[1])) 
            # a2. 生成假样本
            z = torch.randn(param.batch_size, param.gan_dim_latent).to(param.device)
            fake_data = G(real_data_label,z)
            # 判别器含卷积操作，需要三维数据，故假样本二维转三维
            fake_data = torch.reshape(fake_data,(fake_data.shape[0],1,fake_data.shape[1]))
            # a3. 生成真假标签，noise labeling
            real_label = (torch.rand(param.batch_size)/5+0.9).to(param.device)
            fake_label = (torch.rand(param.batch_size)/5+0.1).to(param.device)
            real_label = real_label.reshape((-1,1))
            fake_label = fake_label.reshape((-1,1))
            # a4. 判别器进行预测
            real_pred = D(real_data_label,real_data)
            fake_pred = D(real_data_label,fake_data)
            # a5. 计算损失
            d_loss_real = -torch.mean(real_pred)
            d_loss_fake = -torch.mean(fake_pred)
            #d_loss_real = loss_fn(real_pred, real_label)
            #d_loss_fake = loss_fn(fake_pred, fake_label)
            
            d_loss_grad = compute_gradient_penalty(param,real_data_label,D,real_data,fake_data)
            d_loss = d_loss_real - d_loss_fake + param.lamda*d_loss_grad
            # a6. 训练判别器
            d_optimizer.zero_grad() # 梯度清零
            d_loss.backward(retain_graph=True) # 反向传播计算梯度
            d_optimizer.step() # 更新参数
           # for p in D.parameters():
           #     p.data.clamp_(-0.01, 0.01)

        # b. 训练生成器
        # b1. 生成假样本
        z = torch.randn(param.batch_size, param.gan_dim_latent).to(param.device)
        fake_data = G(real_data_label,z)
        # 判别器含卷积操作，需要三维数据，故假样本二维转三维
        fake_data = torch.reshape(fake_data,(fake_data.shape[0],1,fake_data.shape[1]))
        # b2. 判别器进行预测
        fake_pred = D(real_data_label,fake_data)
        # b3. 计算损失
        # 不做noise labeling
        real_label = torch.ones(param.batch_size,1).to(param.device)
        g_loss = -torch.mean(fake_pred)
        #g_loss = loss_fn(fake_pred, real_label)
        # b4. 训练生成器
        g_optimizer.zero_grad() # 梯度清零
        g_loss.backward() # 反向传播计算梯度
        g_optimizer.step() # 更新参数

        # c. 记录结果并打印
        if mute == False:
            res['d_loss'][epoch] = d_loss.item()
            res['g_loss'][epoch] = g_loss.item()
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '.
                format(epoch+1, param.num_epochs, 
                        d_loss.item(), g_loss.item()))

    # 5. 绘图和保存
    if plot:
        plot_save_train(param,D,G,res)
    # 返回生成器和结果
    return G, res


#%% GAN生成序列
def simu_gan(param,G,train_data,plot):
    """
    返回GAN生成数据

    Parameters
    ----------
    param : Class(Param)
        参数类.
    G : torch.nn.Module的子类
        生成器.
    train_data : T*1 ndarray, optional
        收益率或价格序列. The default is None.

    Returns
    -------
    fake_data : N*seq_lengths ndarray
        若param.use_ret为True，返回收益率.
        若param.use_ret为False，返回标准化价格.
    """
    # 0. 设置随机数种子点
    set_random_seed(param.seed)
    # 测试模式
    G.eval()
    
    # 1. 生成num_gen条假样本
    seq_lengths = param.gan_dim_latent
    real_label=train_data[-param.label_size:]*param.scale_ret
    z = torch.randn(param.num_gen, seq_lengths).to(param.device)
    real_label=torch.Tensor(real_label).repeat(param.num_gen, 1).to(param.device)
    fake_data = G(real_label,z).detach().cpu().numpy()
    
    # 2. 数据处理
    if param.use_ret:
        # 若数据为收益率，缩小一定倍数
        fake_data = fake_data / param.scale_ret
    else:
        # 若数据为价格，直接返回标准化价格
        pass

    # 3. 绘制生成结果
    if plot:
        plot_simu(param,fake_data,train_data)
        plot_monte(param,fake_data,train_data)
    
    # 返回生成数据
    return fake_data

