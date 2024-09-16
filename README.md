# SO-FCE: Self-Optimized Fuzzy Comprehensive Evaluation Method for Addressing Real-World Low-Quality Samples in WebRobot Detection 
### Description
This repository is the open source code for our paper entitled "SO-FCE: Self-Optimized Fuzzy Comprehensive Evaluation Method for Addressing Real-World Low-Quality Samples in WebRobot Detection", which is under review in Applied Soft Computing .

We proposes a Web Robot behavior detection technique based on the self-optimizing fuzzy comprehensive evaluation method. The method provides an in-depth analysis of Web Robot's access behavior patterns from the session perspective,
and combines the fuzzy comprehensive evaluation method and iterative self-optimization learning to optimize the fuzzy decision-making parameters and improve the accuracy and robustness of the detection.

### Dataset
The web access data used in this study comes from the network server area traffic of a university. We extracted HTTP access log data from the network traffic.
Each record contains detailed information such as the source IP, destination IP, source port, destination port, Host, Url, UserAgent, Referer, and access timestamp of the HTTP access.

### How to run the code?

The user guide is presented as follow.

-------------------------------------

#### 1. Data preprocessing
##### a. Download the web session record `WebSessionAccessMain2_GenSample_src_data.bt3.txt` to the `./data` folder.

##### b. Run `pre_process.py`. The training set (`X_train.txt`, `y_train.txt`) and test set (`X_test.txt`, `y_test.txt`) will be generated in the `./data` directory.
```
commond: python pre_process.py
```
##### c. Run `false_samples.py`. Different proportions of low-quality training sets (containing incorrect labels) will be generated in the `./data` directory.

```
commond: python false_sample.py
```
#### 2.Run comparative experiments
##### a.First, edit `train.py`, select the training samples, and run the code. The trained models will be saved in the `models` folder.
Tips: Edit the parameters of `pd.read_csv()` to select the training samples.
```
commond: python train.py
```
##### b. Then run `predict.py` to obtain the results of the comparative experiments.
```
commond: python predict.py
```
#### 3. Run the SO-FCE method
##### Edit `main.py`, select the training samples, and run the program to obtain the results.

```
commond: python main.py
```
