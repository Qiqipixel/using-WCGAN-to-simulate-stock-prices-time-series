# -*- coding: utf-8 -*-
"""
CGAN生成模型核心函数
运行环境：
    windows 10 64bit
    python 3.7
    anaconda >=2019.10
    numpy >=1.18.1
    pandas >=0.25.1
    torch 1.4.0
函数说明：
    make_folders 创建保存模型和结果文件夹
    set_random_seed 设置随机数种子
    read_raw_data 读取原始数据
    extract_train_data 提取训练数据
    subsequences 生成子序列函数
    get_loader_cgan 样本生成器
    write_excel 输出结果   
    extract_train_data_trade取回测的data 
    
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import param_gan as param
import torch
import pickle


#%% 创建保存模型和结果文件夹
def make_folders(param):
    """
    创建保存模型和结果文件夹
    
    Parameters
    ----------
    param : Class(Param)
        参数类.

    Returns
    -------
    None.

    """
    # 创建保存模型文件夹
    if not os.path.exists(param.path_models):
        os.makedirs(param.path_models)
    # 创建保存结果文件夹
    if not os.path.exists(param.path_results):
        os.makedirs(param.path_results)
    # 无返回值
    return None


#%% 设置随机数种子
def set_random_seed(seed):
    """
    设置随机数种子
    
    Parameters
    ----------
    seed : int
        随机数种子.

    Returns
    -------
    None.

    """
    # Set the random seed manually for reproducibility.
    np.random.seed(seed) # numpy种子点
    torch.manual_seed(seed) # torch种子点
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # cuda种子点
    # 无返回值
    return None
                

#%% 提取训练数据
def extract_train_data(raw_data):
    # 价格转换为对数收益
    if param.use_ret:
        train_data = np.log(raw_data[1:] / raw_data[:-1]).astype('float32')
    # 返回训练数据
    return train_data


    """
    提取训练开始和结束日期之间的训练数据

    Parameters
    ----------
    param : Class(Param)
        参数类.
    raw_data : T*N ndarray(时间*指标)
        原始数据.

    Returns
    -------
    train_data : T*1 ndarray(时间*指标)
        若param.use_ret为True，返回T*1收益率序列.
        若param.use_ret为False，返回T*1价格序列.

    """
    # 确定训练开始结束日期索引
    idx_start = raw_data.index.tolist().index(start_day)
    idx_end = raw_data.index.tolist().index(end_day)
    
    # 如果训练开始日期为空值，从第一个非nan元素开始
    # if np.isnan(raw_data.iloc[idx_start,param.use_col]):
    #     idx_start = np.argwhere(np.isnan(raw_data.iloc[:,param.use_col]).values==False)[0][0]
    # 提取训练数据，格式转换成ndarray
    rawdataframe=raw_data.iloc[idx_start:idx_end+1]
    train_data = raw_data.iloc[idx_start:idx_end+1].values
    
    # 价格转换为对数收益
    if param.use_ret:
        train_data = np.log(train_data[1:] / train_data[:-1]).astype('float32')
    # 返回训练数据
    return train_data

#%% 生成label+real的股票序列

def labelreal(data,label_size,real_size):
    """
    对原始序列data，连续取长度为label_size的子序列和接下来real_size的子序列

    Parameters
    ----------
    data : 1d ndarray
        原始序列.
    label_size : int
        子序列长度.
    real_size : int
        条件序列长度
    Returns
    -------
    N*T ndarray
        子序列.
    
    Example
    -------
    labelreal(np.array([1,2,3,4,5,6]),3,2)
    return [array([[1, 2, 3],
                  [2, 3, 4]]),
            array([[4, 5],
                   [5, 6]])]
    """
    n=data.shape[0]
    indice_lable= np.arange(label_size) + np.arange(n-label_size-real_size+1).reshape(-1, 1)
    indice_real = np.arange(label_size,label_size+real_size)+ np.arange(n-label_size-real_size+1).reshape(-1, 1)
    return [data[indice_lable],data[indice_real]]


#%% 样本生成器
def get_loader_cgan(param,data,window_width_lable,window_width_real,batch_size,shuffle=False):
    """
    返回样本生成器

    Parameters
    ----------
    param : Class(Param)
        参数类.
    data : T*1 ndarray
        收益率或价格序列.
    window_width : int
        子序列长度.
    batch_size : int
        batch样本数.
    shuffle : bool, optional
        是否打乱. The default is True.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        样本生成器，生成batch_size*seq_lengths矩阵.
        若param.use_ret为True，样本为收益率，乘以param.scale_ret倍（默认10倍）.
        若param.use_ret为False，样本为价格，每条子序列标准化.
    """
    # 截取长度为window_width的子序列
    samples_label = labelreal(data, window_width_lable,window_width_real)[0].astype(np.float32)
    samples_real  = labelreal(data, window_width_lable,window_width_real)[1].astype(np.float32)
    # 数据预处理
    if param.use_ret:
        # 若输入数据为收益率，乘以param.scale_ret倍（默认10倍）
        samples_label = samples_label * param.scale_ret
        samples_real = samples_real * param.scale_ret
    else:
        # 若输入数据为价格，每条子序列标准化
        samples_label = (samples_label - samples_label.mean(axis=1, keepdims=True)) / samples_label.std(axis=1, keepdims=True)
        samples_real = (samples_real - samples_real.mean(axis=1, keepdims=True)) / samples_real.std(axis=1, keepdims=True)
    # 若batch_size大于样本数量，batch_size设为样本数量
    if batch_size > samples_label.shape[0]:
        batch_size = samples_label.shape[0]
    # 合成标签和真实数据
    # DataLoader，batch数据生成器，类似Keras中fit_generator的输入，不需要一次性读取训练数据，减少内存占用
    loader_label = torch.utils.data.DataLoader(samples_label, 
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    loader_real = torch.utils.data.DataLoader(samples_real, 
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    # 返回样本生成器
    return loader_label,loader_real##这里我返回了两个迭代器，一个是标签的迭代器，一个是真实数据的迭代器

def get_loader_cgan_2(param,data,window_width_lable,window_width_real,batch_size,shuffle=False):
    # 截取长度为window_width的子序列
    samples_label = labelreal(data, window_width_lable,window_width_real)[0].astype(np.float32)
    samples_real  = labelreal(data, window_width_lable,window_width_real)[1].astype(np.float32)
    # 数据预处理
    if param.use_ret:
        # 若输入数据为收益率，乘以param.scale_ret倍（默认10倍）
        samples_label = samples_label * param.scale_ret
        samples_real = samples_real * param.scale_ret
    else:
        # 若输入数据为价格，每条子序列标准化
        samples_label = (samples_label - samples_label.mean(axis=1, keepdims=True)) / samples_label.std(axis=1, keepdims=True)
        samples_real = (samples_real - samples_real.mean(axis=1, keepdims=True)) / samples_real.std(axis=1, keepdims=True)
    # 若batch_size大于样本数量，batch_size设为样本数量
    if batch_size > samples_label.shape[0]:
        batch_size = samples_label.shape[0]
    # 合成标签和真实数据
    samples= zip(samples_label,samples_real)
    #DataLoader，batch数据生成器，类似Keras中fit_generator的输入，不需要一次性读取训练数据，减少内存占用
    loader = torch.utils.data.DataLoader(list(samples), 
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    # 返回样本生成器###这里只有一个是很方便的存在，实现了在时间序列的可shuffle性
    return loader
#%% 绘制和保存训练结果
def plot_save_train(param,D,G,res):
    """
    绘制和保存训练结果

    Parameters
    ----------
    param : Class(Param)
        参数类.
    D : torch.nn.Module的子类
        判别器.
    G : torch.nn.Module的子类
        生成器.
    res : num_epochs*2 DataFrame
        d_loss和g_loss.

    Returns
    -------
    None.

    """
    # 绘制损失图
    plt.figure()
    plt.plot(res['d_loss'])
    plt.plot(res['g_loss'])
    plt.legend(['d_loss','g_loss'])
    plt.show()
    
    
    return None


#%% 绘制生成结果
def plot_simu(param,fake_data,train_data=None):
    """
    绘制生成结果

    Parameters
    ----------
    param : Class(Param)
        参数类.
    fake_data : N*T ndarray
        虚假收益率序列.
    train_data : T*1 ndarray, optional
        真实价格或收益率序列. The default is None.

    Returns
    -------
    None.

    """
    # 绘制真样本
    if not train_data is None:
        plt.figure()
        if param.use_ret:
            plt.plot(np.exp(train_data.cumsum()))
        else:
            plt.plot(train_data)
        plt.show()

    # 绘制前30条假样本
    if fake_data.shape[0] >= 30:
        plt.figure()
        for i in range(30):
            plt.subplot(5,6,i+1)
            if param.use_ret:
                plt.plot(np.exp(fake_data[i,:].cumsum()))
            else:
                plt.plot(fake_data[i,:])
        plt.show()
        
    return None

def plot_monte(param,fake_data,train_data=None):
    """
    绘制生成结果

    Parameters
    ----------
    param : Class(Param)
        参数类.
    fake_data : N*T ndarray
        虚假收益率序列.
    train_data : T*1 ndarray, optional
        真实价格或收益率序列. The default is None.

    Returns
    -------
    None.

    """
    # 绘制真样本
    fake_sum=np.exp(np.cumsum(fake_data, axis=1))
    result = np.concatenate((np.ones((param.num_gen, 1)),fake_sum), axis=1)
    col = np.empty((param.num_gen, param.label_size-1))
    col[:] = np.nan
    result_co = np.concatenate((col, result), axis=1)
    plt.plot(np.exp(train_data.cumsum()[-param.label_size:]))
    plt.plot(result_co.T*np.exp(train_data.cumsum()[-1:]),alpha=0.4,linestyle='--')
    plt.plot((result_co*np.exp(train_data.cumsum()[-1:])).mean(axis=0), color='black', linewidth=2) 
        
    return None

#%% 输出结果
def write_excel(param,res_train,train_data,fake_data_gan):
    """
    输出结果

    Parameters
    ----------
    param : Class(Param)
        参数类.
    res_train : num_epochs*2 DataFrame
        d_loss和g_loss.
    train_data : T*1 ndarray
        真实价格或收益率序列.
    fake_data_gan : N*T ndarray
        GAN生成的虚假收益率序列.
    Returns
    -------
    None.

    """
    # 1. 训练结果
    output_res_train = pd.DataFrame(res_train)
    
    # 2. 真样本，GAN生成样本
    # 收益转换为价格
    if param.use_ret:
        train_data = np.exp(train_data.cumsum())
        fake_data_gan = np.exp(fake_data_gan.cumsum(axis=1))
    # ndarray转换为DataFrame
    output_real_data = pd.DataFrame(train_data)
    output_fake_data_gan = pd.DataFrame(fake_data_gan)
    
    # 3. 参数
    output_model_params = dict()
    # 遍历param的属性
    for i_key in dir(param):
        if i_key[:2]!='__': # 排除内置函数
            # 提取参数值
            output_model_params[i_key] = eval('param.'+i_key)
    # 格式字典转为Series
    output_model_params = pd.Series(output_model_params)
    
    # 统一输出到一个Excel文件
    print('start writing excel')
    if param.path_suffix=='': # 不设置后缀名
        writer = pd.ExcelWriter(param.path_results+'results'+param.gan_type+'.xlsx')
    else: # 设置后缀名
        writer = pd.ExcelWriter(param.path_results+'results_'+param.gan_type+'_'+param.path_suffix+'.xlsx')
    output_res_train.to_excel(writer,sheet_name='res_train')
    output_real_data.to_excel(writer,sheet_name='real_data')
    output_fake_data_gan.to_excel(writer,sheet_name='fake_data_'+param.gan_type)
    output_model_params.to_excel(writer,sheet_name='params')
    writer.close()
    print('finish writing excel')        
    

#%% 调试程序时使用
if __name__ == '__main__':
    # 读取参数
    import param_gan as param  
    # 创建路径
    make_folders(param)
    # 读取原始数据
    raw_data = read_raw_data(param)
    # 提取训练数据：t*1价格
    train_data = extract_train_data(param,raw_data)
    
    if param.gan_type=='gan':
        import core_gan
        # 构建并训练GAN
        G, res_train = core_gan.train_gan(param,train_data)
        # 使用GAN生成数据
        fake_data_gan = core_gan.simu_gan(param,G,train_data=train_data)
            
    # Bootstrap生成序列（对照组）
    fake_data_bs = np.zeros(fake_data_gan.shape)
    # GARCH模型生成序列（对照组）
    fake_data_garch = np.zeros(fake_data_gan.shape) # ctrl_garch(param,train_data)
    # 结果输出至excel
    write_excel(param,res_train,train_data,fake_data_gan,fake_data_bs,fake_data_garch)
    
#%% fake_data feature
def auto_corr(fakedata,real_data,plot=True,lag=param.gan_dim_output):
    import statsmodels.api as sm
    acf_real=sm.tsa.stattools.acf(real_data, nlags=lag, fft=False)
    if plot:
        plt.scatter(np.arange(lag), acf_real[1:], s=5)
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=-1.96/np.sqrt(len(acf_real)), color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=1.96/np.sqrt(len(acf_real)), color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Lag')
        plt.ylim(-1, 1)
        plt.ylabel('Autocorrelation Coefficient')
        plt.title('n-order Autocorrelation Real Data')
        plt.show()
        if fakedata.shape[0] >= 30:
            plt.figure()
            for i in range(30):
                plt.subplot(5,6,i+1)
                acf = sm.tsa.stattools.acf(fakedata[i], nlags=lag-1, fft=False)
                plt.scatter(np.arange(lag-1), acf[1:], s=1)
                plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                plt.axhline(y=-1.96/np.sqrt(len(fakedata[i])), color='gray', linestyle='--', linewidth=1)
                plt.axhline(y=1.96/np.sqrt(len(fakedata[i])), color='gray', linestyle='--', linewidth=1)
                plt.ylim(-1, 1)
            plt.show()
    return None

#%% fake_data feature
def abs_auto_corr(fakedata,real_data,plot=True,lag=param.gan_dim_output):
    import statsmodels.api as sm
    acf_real=sm.tsa.stattools.acf(real_data**2, nlags=lag, fft=False)
    if plot:
        plt.scatter(np.arange(lag), acf_real[1:], s=5)
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=-1.96/np.sqrt(len(acf_real)), color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=1.96/np.sqrt(len(acf_real)), color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Lag')
        plt.ylim(-1, 1)
        plt.ylabel('Autocorrelation Coefficient')
        plt.title('n-order Autocorrelation Real Data')
        plt.show()
        if fakedata.shape[0] >= 30:
            plt.figure()
            for i in range(30):
                plt.subplot(5,6,i+1)
                acf = sm.tsa.stattools.acf(fakedata[i]**2, nlags=lag-1, fft=False)
                plt.scatter(np.arange(lag-1), acf[1:], s=1)
                plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                plt.axhline(y=-1.96/np.sqrt(len(fakedata[i])), color='gray', linestyle='--', linewidth=1)
                plt.axhline(y=1.96/np.sqrt(len(fakedata[i])), color='gray', linestyle='--', linewidth=1)
                plt.ylim(-1, 1)
            plt.show()
    return None
#%%
def plot_stra(raw_data_close,stock_index):
    raw_data_close['rate']=raw_data_close['close'].diff().shift(-1)#计算价差
    raw_data_close['ind'] = np.where(raw_data_close['trade'] & raw_data_close['change'], True,
                            np.where(raw_data_close['trade'] | raw_data_close['change'], False, np.nan))#生成空仓数据
    raw_data_close=raw_data_close.fillna(method='ffill')
    raw_data_close['rate_trade']=raw_data_close['ind']*raw_data_close['rate']#计算空仓收益
    raw_data_close['trade_price']=(raw_data_close['rate_trade'].cumsum()+raw_data_close['close'].values[0]).shift(1)
    raw_data_close['date']=pd.to_datetime(raw_data_close.date)#转换日期
    zero_mask = (raw_data_close['rate_trade'] == 0) & (raw_data_close['ind'] == 0)
    zero_starts = raw_data_close.loc[zero_mask & ~zero_mask.shift(1).fillna(False)].iloc[:, 0]
    zero_ends = raw_data_close.loc[zero_mask & ~zero_mask.shift(-1).fillna(False)].iloc[:, 0]
    plt.plot(raw_data_close['date'],raw_data_close['trade_price'],label='time_select')#画图
    plt.plot(raw_data_close['date'],raw_data_close['close'],label='real')
    for star, end in zip(zero_starts,zero_ends):
        plt.axvspan(star,end,alpha=0.5, color='grey')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Trade Strategy based on {}'.format(stock_index))
    plt.legend()
    plt.savefig('figure_{}.png'.format(stock_index))
    plt.show()
    return None





    