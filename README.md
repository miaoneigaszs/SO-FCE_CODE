# SO-FCE: Self-Optimized Fuzzy Comprehensive Evaluation Method for Addressing Real-World Low-Quality Samples in WebRobot Detection 
### Description
This repository is the open source code for our paper entitled "SO-FCE: Self-Optimized Fuzzy Comprehensive Evaluation Method for Addressing Real-World Low-Quality Samples in WebRobot Detection", which is under review in IEEE xxxx.

We proposes a Web Robot behavior detection technique based on the self-optimizing fuzzy comprehensive evaluation method. The method provides an in-depth analysis of Web Robot's access behavior patterns from the session perspective,
and combines the fuzzy comprehensive evaluation method and iterative self-optimization learning to optimize the fuzzy decision-making parameters and improve the accuracy and robustness of the detection.

### Dataset
The web access data used in this study comes from the network server area traffic of a university. We extracted HTTP access log data from the network traffic.
Each record contains detailed information such as the source IP, destination IP, source port, destination port, Host, Url, UserAgent, Referer, and access timestamp of the HTTP access.

### How to run the code?

The user guide is presented as follow.

-------------------------------------

#### 1.Data preprocessing
##### a.下载web会话记录 WebSessionAccessMain2_GenSample_src_data.bt3.txt 到./data文件夹下.

##### b.运行pre_process.py，在./data目录下会生成训练集（X_train.txt, y_train.txt）和测试集(X_test.txt, y_test.txt).
```
commond: python pre_process.py
```
##### c.运行false_samples.py, 在./data目录下会生成不同比例的低质量训练集（含有错误标注）.
```
commond: python false_sample.py
```
#### 2.运行对比实验
##### a.首先编辑train.py, 选择训练样本, 运行代码, 训练好的模型将保存在在models文件夹中.
Tips: 通过编辑'pd.read_csv()'的参数, 来选择训练样本.
```
commond: python train.py
```
##### b.然后运行predict.py，获得对比实验的运行结果
```
commond: python predict.py
```
#### 3.运行SO-FCE方法
##### 编辑main.py, 选择训练样本, 运行该程序, 获得运行结果
```
commond: python main.py
```
