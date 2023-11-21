#%%
import numpy as np
from Gan_Simulator import *
import param_gan as param
from tradeindx import *
#%%
F=open(r'C:\Users\81913\Desktop\Cgan\Code\stock_price_standard','rb')
content=pickle.load(F)
stock_batch=get_stock_batch(content)
windows=param.windows
#%%
'''
运行以下代码会从选定回测期的数据集中随机抽取一个股票进行回测
'''
for stock_index in stock_batch.sample(n=1):
    raw_data=get_raw_data(content,stock_index)
    raw_data_close=get_raw_data_close(raw_data,windows)
    for i in range(0,len(raw_data)-windows-1,param.freq):
        start_day=raw_data.index[i]
        end_date=raw_data.index[i+windows]
        train_data=extract_train_data_trade(start_day,end_date,raw_data).reshape(-1)
        if param.gan_type=='gan':
            import core_cwgan_gp_v3
            G,res_train = core_cwgan_gp_v3.train_gan(param,train_data,mute=True,plot=False)
            fake_data_gan = core_cwgan_gp_v3.simu_gan(param,G,train_data=train_data,plot=False)
        
        fake_sum=np.exp(np.cumsum(fake_data_gan, axis=1))
        result = np.concatenate((np.ones((param.num_gen, 1)),fake_sum), axis=1)*np.exp(train_data.cumsum()[-1:])
        
        if trade_or_not(result):
            raw_data_close.loc[raw_data_close['date'] == end_date, 'trade'] = True
        
        raw_data_close.loc[raw_data_close['date'] == end_date, 'change'] = True
        print(end_date,trade_or_not(result))
    plot_stra(raw_data_close,stock_index)




