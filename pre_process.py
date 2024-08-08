import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv("data/WebSessionAccessMain2_GenSample_src_data.bt3.txt", delimiter='\t', header=None)
# Add column names to the data
column_names = ['id', 'cookie_ratio', 'refer_ratio', 'js_ratio', 'res_num', 'res_cplx', 'inter_avg', 'inter_var', 'depth_var',
                'dup_ratio', 'cross_vst', 'robotcnt', 'spidercnt', 'web1', 'web2', 'web3', 'web4', 'web5', 'label']
data.columns = column_names
# Extract IP addresses from the id column
data['ip'] = data['id'].apply(lambda x: '.'.join(x.split('-')[-1].split('.')[-4:]))

ip_info_file = 'data/ip_info_32.txt'
ip_info = pd.read_csv(ip_info_file, header=None, names=['ip', 'from'])
# Merge IP address information with the dataset
merged_df = pd.merge(data, ip_info, on='ip', how='left')

# Label empty values in the 'from' column as '-'
merged_df['from'] = merged_df['from'].fillna('-').apply(lambda x: '-' if x == '-' else x)
data = merged_df

# Read IP address information, file format is "220.181.51,baidu"
ip_info_file = 'data/ip_info_24.txt'
ip_info = pd.read_csv(ip_info_file, header=None, names=['ip_prefix', 'from1'])

# Extract the first 24 bits as a prefix
data['ip_prefix'] = data['ip'].apply(lambda x: '.'.join(x.split('.')[:3]))
# Merge the first 24-bit IP address information with the dataset
merged_df = pd.merge(data, ip_info, on='ip_prefix', how='left')

# Label empty values in the 'from1' column as '-'
merged_df['from1'] = merged_df['from1'].fillna('-').apply(lambda x: '-' if x == '-' else x)

# Drop the temporary prefix column
merged_df.drop(['ip_prefix'], axis=1, inplace=True, errors='ignore')

# Combine information from both columns
merged_df['final_from'] = merged_df['from'].where(merged_df['from'] != '-', merged_df['from1'])

# Drop the 'from' and 'from1' columns
merged_df.drop(['from', 'from1'], axis=1, inplace=True)

data = merged_df

# Statistics on IP segments
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
# Retain rows where label=1 and from is not '-'
black_samples = test[(test['label'] == 1) & (test['final_from'] != '-')]

# Retain rows where label=0 and IP address prefix is '10.132'
prefix_condition = test['ip_16'].str.startswith('10.132')
white_samples = test[(test['label'] == 0) & prefix_condition]

# Determine the column name to operate on
column_name = 'ip'

# Calculate the number of rows to delete
original_length = len(white_samples)
rows_to_remove = original_length // 3 * 2

# Randomly delete rows
random_rows = white_samples.sample(n=rows_to_remove)
white_samples = white_samples.drop(random_rows.index)

# Merge rows that meet the conditions
test_data = pd.concat([black_samples, white_samples])

test_data.drop(['ip_16', 'ip'], axis=1, inplace=True)

total_num = test_data.shape[0]
# Count the number of rows where 'label' column is 0
white = test_data[test_data['label'] == 0].shape[0]
# Count the number of rows where 'label' column is 1
black = test_data[test_data['label'] == 1].shape[0]

print(f"Total samples: {total_num}")
print(f"Total white samples with label '0': {white}, proportion: {white / total_num}")
print(f"Total black samples with label '1': {black}, proportion: {black / total_num}")
print(f"White to black sample ratio: {white/black}")

print(test_data.columns)
# Drop columns that are temporarily not needed
test_data.drop(['id', 'web1', 'web2', 'web3', 'web4', 'web5', 'final_from', 'robotcnt', 'spidercnt'], axis=1, inplace=True)

feature_columns = ['cookie_ratio', 'refer_ratio', 'js_ratio', 'res_num', 'res_cplx', 'inter_avg', 'inter_var', 'depth_var',
                'dup_ratio', 'cross_vst']

# Check for missing values
for feature_column in feature_columns:
    if test_data[feature_column].isnull().values.any():
        print(f"Missing values found in dataset {feature_column}. Total missing: ")
        print(test_data[feature_column].isnull().sum())
    else:
        print(f"No missing values in dataset {feature_column}.")

# Drop rows with missing values
print('')
test_data = test_data.dropna(how='any')
for feature_column in feature_columns:
    if test_data[feature_column].isnull().values.any():
        print(f"Missing values found in dataset {feature_column}. Total missing: ")
        print(test_data[feature_column].isnull().sum())
    else:
        print(f"No missing values in dataset {feature_column}.")

# X are features, y is the target variable 'label'
X = test_data.drop(columns=['label'])
y = test_data['label']

# Randomly split the dataset, test_size indicates the proportion of the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Output the sizes of the split datasets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

# Save to txt files
X_test.to_csv("data/X_test.txt", sep='\t', index=False)
X_train.to_csv("data/X_train.txt", sep='\t', index=False)
y_test.to_csv("data/y_test.txt", sep='\t', index=False)
y_train.to_csv("data/y_train.txt", sep='\t', index=False)
