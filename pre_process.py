import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv(
    "data/WebSessionAccessMain2_GenSample_src_data.bt3.txt", delimiter='\t', header=None)
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

# 在 'from' 列下面写上标注，并将空值标注为'-'
merged_df['from'] = merged_df['from'].fillna('-').apply(lambda x: '-' if x == '-' else x)
data = merged_df
# 显示修改后的 DataFrame
#print(data)

# 读取 IP 地址对应信息，文件格式为 "220.181.51,baidu"
ip_info_file = 'data/ip_info_24.txt'
ip_info = pd.read_csv(ip_info_file, header=None, names=['ip_prefix', 'from1'])

# 提取前24位作为前缀
data['ip_prefix'] = data['ip'].apply(lambda x: '.'.join(x.split('.')[:3]))
# 将前24位 IP 地址对应信息与数据集合并
merged_df = pd.merge(data, ip_info, on='ip_prefix', how='left')

# 在 'from' 列下面写上标注
merged_df['from1'] = merged_df['from1'].fillna('-').apply(lambda x: '-' if x == '-' else x)

# 删除临时的前缀列
merged_df.drop(['ip_prefix'], axis=1, inplace=True, errors='ignore')

# 合并两列信息
merged_df['final_from'] = merged_df['from'].where(merged_df['from'] != '-', merged_df['from1'])

# 删除 'from' 和 'from1' 列
merged_df.drop(['from', 'from1'], axis=1, inplace=True)

# 显示修改后的 DataFrame
data = merged_df
#print(data)

#统计一下ip的分段
white_dash = data[['ip']].copy()
white_dash['ip_16'] = white_dash['ip'].apply(lambda x: '.'.join(x.split('.')[:2]))

white_dash = white_dash.sort_values(by='ip_16', ascending=False)

white_dash.drop(['ip'], axis=1, inplace=True)
grouped_ip = white_dash.groupby('ip_16')
counts_ip = grouped_ip.size().reset_index(name='cnt')
sorted_counts_ip = counts_ip.sort_values(by='cnt', ascending=False)

print(sorted_counts_ip)


test = data
test['ip_16'] = test['ip'].apply(lambda x: '.'.join(x.split('.')[:2]))
# 保留条件：label=1 且 from 不等于 '-' 的行
black_samples = test[(test['label'] == 1) & (test['final_from'] != '-')]

# 保留条件：label=0 且 ip地址前缀为‘10.132’的行
prefix_condition = test['ip_16'].str.startswith('10.132')
white_samples = test[(test['label'] == 0) & prefix_condition]


# 确定要操作的列名
column_name = 'ip'

# 计算要删除的行数量
original_length = len(white_samples)
rows_to_remove = original_length // 3 * 2

# 随机删除行
random_rows = white_samples.sample(n=rows_to_remove)
white_samples = white_samples.drop(random_rows.index)

# 合并满足条件的行
test_data = pd.concat([black_samples, white_samples])

test_data.drop(['ip_16', 'ip'], axis = 1, inplace = True)

total_num = test_data.shape[0]
#统计整体中 'label' 列为 0 的行数
white = test_data[test_data['label'] == 0].shape[0]
#统计整体中 'label' 列为 1 的行数
black = test_data[test_data['label'] == 1].shape[0]

print(f"样本总数: {total_num}")
print(f"整体label为 '0' 的白样本数: {white}，占比{white / total_num}")
print(f"整体label为 '1' 的黑样本数: {black}，占比{black / total_num}")
print(f"白黑样本比例：{white/black}")

print(test_data.columns)
#删除暂时不需要的列
test_data.drop(['id', 'web1', 'web2', 'web3', 'web4', 'web5', 'final_from', 'robotcnt', 'spidercnt'], axis=1, inplace=True)

feature_columns = ['cookie_ratio', 'refer_ratio', 'js_ratio', 'res_num', 'res_cplx', 'inter_avg', 'inter_var', 'depth_var',
                'dup_ratio', 'cross_vst']

#检查缺失值
for feature_column in feature_columns:
    if test_data[feature_column].isnull().values.any():
        print(f"数据集{feature_column}中存在缺失值.共缺失：")
        print(test_data[feature_column].isnull().sum())
    else:
        print(f"数据集{feature_column}中没有缺失值.")
#删除缺失值
print('')
test_data = test_data.dropna(how='any')
for feature_column in feature_columns:
    if test_data[feature_column].isnull().values.any():
        print(f"数据集{feature_column}中存在缺失值.共缺失：")
        print(test_data[feature_column].isnull().sum())
    else:
        print(f"数据集{feature_column}中没有缺失值.")


# X 特征，y 是目标变量 'label'
X = test_data.drop(columns=['label'])
y = test_data['label']

# 随机分割数据集，test_size 表示测试集的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 输出分割后的数据集大小
print("训练集大小:", len(X_train))
print("测试集大小:", len(X_test))

#保存到txt文件中
# X_test.to_csv("data\X_test.txt", sep='\t', index=False)
# X_train.to_csv("data\X_train.txt", sep='\t', index=False)
# y_test.to_csv("data\y_test.txt", sep='\t', index=False)
# y_train.to_csv("data\y_train.txt", sep='\t', index=False)