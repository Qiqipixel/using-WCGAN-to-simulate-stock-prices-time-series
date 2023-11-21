# using-WCGAN-to-simulate-stock-prices-time-series
Do you want use some historical data to predict the price in the a short future? Then this is what are you looking for. In this model you can use WCGAN to learn the feature between the historical data and the future data, then you can input a series data to output a bunch of future data.


core_cwgan_gp_v3.py 包含生成器与判别器何模型的训练\\
Gan_Simulator.py 包含数据的处理与生成\\
param_gan.py 包含训练所需的所有参数\\
tradeindx.py 包含回测指标的计算数据的提取等\\
main_cgan_diffver包含不同版本模型的训练和结果\\
