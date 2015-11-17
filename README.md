
TSA 
================
Machine learning in time series analysis

programs for test:
test_EKF.py  用来测试新功能块的程序
gramTest.py  用来测试不太确定的python, numpy 或 theano 特性
scan_time.py 用来测试theano中的循环

programs for backup and referring:

simpleRNN.py     一个简单的RNN例子，序列化的处理方式，音调识别的例子
simpleLSTMtmp.py 来在Deeplearing的参考程序
simpleLSTM.py    LSTM的简单例子
simpleFF_s.py    序列化的简单前向网络
simpleFF.py      使用构造数据的前向网络
rnn.py           一个简单的RNN例子， 来自https://github.com/gwtaylor/theano-rnn
blockRNN.py      RNN的分块化实现

adaRNN.py        使用ada训练的RNN
EKF_RNN.py       使用EKF训练的RNN


programs Under development:
PRNN.py      级联的RNN，用于信号实时处理的结构
MaskRNN.py   使用额处掩码的RNN， 使用ada训练 
CW_RNN.py    基于MaskRNN的CW-RNN实现， 使用ada训练 
MaskRNN_EKF  使用EKF训练的MaskRNN

others:
idmb*.py     LSTM例子所需要的数据预处理程序


