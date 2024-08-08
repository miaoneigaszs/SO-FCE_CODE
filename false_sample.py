import pandas as pd
import numpy as np
y_train = pd.read_csv("y_train.txt")
df = pd.read_csv("X_train.txt", delimiter='\t')
df['label'] = y_train['label']
df['sign'] = 0
#设置要改变的掺错率
percentage_to_change = 0.45
num_1 = df[df['label'] == 1].shape[0]
# 找到所有 'label' 列中值为1的索引
indices_to_change = df.index[df['label'] == 1]
# 随机选择要改变的索引
indices_to_change = np.random.choice(indices_to_change, size=int(percentage_to_change * len(indices_to_change)), replace=False)
# 将选择的索引对应的 'label' 值改为0
df.loc[indices_to_change, 'label'] = 0
df.loc[indices_to_change, 'sign'] = 1

# 找到所有 'label' 列中值为1的索引
indices_to_change = df.index[(df['label'] == 0) & (df['sign'] == 0)]
# 随机选择要改变的索引
indices_to_change = np.random.choice(indices_to_change, size=int(percentage_to_change * len(indices_to_change)), replace=False)
# 将选择的索引对应的 'label' 值改为0
df.loc[indices_to_change, 'label'] = 1

y_train = df['label']
df.drop(['label', 'sign'], axis=1, inplace=True)

#保存生成的文件
y_train.to_csv("all_y_train_0.45_wrong.txt", sep='\t', index=False)
df.to_csv("all_X_train_0.45_wrong.txt", sep='\t', index=False)