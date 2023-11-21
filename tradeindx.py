"""
数据与交易指标函数

"""
import numpy as np
import pandas as pd
import param_gan as param
#%%
def sharp_ratio(data):
    returns = np.diff(data) / data[:-1]
    annual_returns = returns.mean() * 252
    # 计算年化波动率
    volatility = returns.std() * np.sqrt(252)
    # 计算夏普比率
    sharpe_ratio = annual_returns / volatility
    return sharpe_ratio
#%%
#取出夏普比率为前一半的样本进行ROC的计算
def select_first_half_indices(seq):
    n = len(seq)
    sorted_indices = sorted(range(n), key=lambda i: seq[i],reverse=True)
    half = n // 2
    return sorted_indices[:half]
#%%是否持有或者空仓
def trade_or_not(result):
    sharp=pd.DataFrame(result[0:7]).apply( lambda x: sharp_ratio(x),axis=1)
    res_roc=result[select_first_half_indices(sharp)]
    ROC=res_roc[:,-1]-res_roc[:,0]
    ind=ROC[ROC>0].shape[0]/(ROC>0).shape[0]
    if ind>0.5:
        res=True
    else:
        res=False
    return res
#%%
#从回测期内选择股票
def get_stock_batch(content):
    content1=content[pd.to_datetime(param.start)].dropna().reset_index().rename(columns={'level_0':'stock_index'}).drop_duplicates(subset=['stock_index'])
    content2=content[pd.to_datetime(param.end)].dropna().reset_index().rename(columns={'level_0':'stock_index'}).drop_duplicates(subset=['stock_index'])
    merge=pd.merge(content1,content2,left_on='stock_index',right_on='stock_index',how='inner')
    return merge.stock_index
#输入生成数据
def get_raw_data(content,stock_index):
    raw_data=content.loc[(stock_index,'Close')]
    raw_data.index = [i.strftime('%Y/%m/%d') for i in raw_data.index.tolist()]
    temp=raw_data.reset_index()
    temp.columns = ['_'.join(col) for col in temp.columns]
    raw_data=temp.rename(columns={temp.columns[0]:'date',temp.columns[1]:'close'}).set_index('date')
    idx_start = raw_data.index.tolist().index(param.start)
    idx_end = raw_data.index.tolist().index(param.end)
    raw_data=raw_data.iloc[idx_start:idx_end+1]
    return raw_data
#得到回测数据
def get_raw_data_close(raw_data,windows):
    raw_data_close=raw_data.reset_index().rename(columns={'index':'date'})
    raw_data_close['trade']=False
    raw_data_close['change']=False
    raw_data_close=raw_data_close.iloc[windows:]
    return raw_data_close

