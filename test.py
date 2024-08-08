import pandas as pd
import numpy as np
data = pd.read_csv(
    'data/WebSessionAccessMain2_GenSample_src_data.bt3.txt', delimiter='\t', header=None)
# 为数据添加列名
column_names = ['id', 'cookie_ratio', 'refer_ratio', 'js_ratio', 'res_num', 'res_cplx', 'inter_avg', 'inter_var', 'depth_var',
                'dup_ratio', 'cross_vst', 'robotcnt', 'spidercnt', 'web1', 'web2', 'web3', 'web4', 'web5', 'label']
data.columns = column_names
#提取id中的ip地址
data['ip'] = data['id'].apply(lambda x: '.'.join(x.split('-')[-1].split('.')[-4:]))
#print(data)

ip_info_file = 'data/ip_info_32.txt'
ip_info = pd.read_csv(ip_info_file, header=None, names=['ip', 'from'])
# 将 IP 地址对应信息与数据集合并
merged_df = pd.merge(data, ip_info, on='ip', how='left')

# print(merged_df)
# 在 'from' 列下面写上标注，并将空值标注为'-'
merged_df['from'] = merged_df['from'].fillna('-').apply(lambda x: '-' if x == '-' else x)
data = merged_df
# 显示修改后的 DataFrame
print(data)