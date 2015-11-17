# -*- coding: utf-8 -*-
"""
使用Elman网络（简单局部回归网络）

@author: simon
"""
import sys,time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from collections import OrderedDict

import utilities.datagenerator as DG

compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def data_get_data_x_y(seq_data, overhead):
    data_x = seq_data[:-1]        # 最后一个不参与
    data_y = seq_data[overhead:]
    return data_x, data_y

def step(*args):
    #global n_input, n_hidden 
    print args
    x =  [args[u] for u in xrange(n_input)] 
    hid_taps = args[n_input]  
    
    W_in =  [args[u] for u in xrange(n_input + 1, n_input * 2 + 1)]
    b_in = args[n_input * 2 + 1]
    W_hid = args[n_input * 2 + 2]
    
    
    h = T.dot(x[0], W_in[0])
    for j in xrange(1, n_input):           # 前向部分
        h +=  T.dot(x[j], W_in[j])
    
    h += T.dot(hid_taps, W_hid)            # 回归部分
    h += b_in                              # 偏置部分

        
    return T.tanh(h)
    
def purelin(*args):
    print args
    h = args[0]
    W_in =  [args[u] for u in xrange(1, n_input + 1)]
    b_in = args[n_input + 1]
    W_hid = args[n_input + 2]
    W_out = args[n_input + 3]
    b_out = args[n_input + 4]

    y = T.dot(h,W_out) + b_out
    return y #T.tanh(y)

def extend_kalman_train(W, y_hat, dim_y_hat, y, x):
    '''
    P: 状态对应的协方差矩阵
    Qv: 观测噪声的协方差矩阵
    Qw: 输入噪声的协方差矩阵
    W: 需要估计的状态
    y_hat: 系统的输出
    dim_y_hat: y_hat 的维数
    y: 期望的输出

    注意：P, Qv, Qw, W的元素个数相同，类型均为list, 对应不同的W
    '''

    # 系数矩阵的向量化
    dim_Wv = 0
    W_vec = []
    for i in numpy.arange(len(W)):
        dim_Wv += W[i].get_value().size
        W_vec.extend([W[i].flatten()])
    W_vec = tuple(W_vec)
    W_vec = T.concatenate(W_vec)  

    print 'number of parameters: ',dim_Wv

    P = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0)  ) # 状态的协方差矩阵
    
    Qw = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0) )  # 输入噪声协方差矩阵， 
    
    Qv = theano.shared( numpy.eye(dim_y_hat) * numpy_floatX(0.01) )  # 观测噪声协方差矩阵

    # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数
    B = []
    for _W in W:
        J, updates = theano.scan(lambda i, y_hat, W: T.grad(y_hat[i], _W).flatten(), 
                                 sequences=T.arange(y_hat.shape[0]), 
                                 non_sequences=[y_hat, _W])
        B.extend([J])

    B = T.concatenate(tuple(B),axis=1)

    # 计算残差
    a = y - y_hat # 单步预测误差

    # 计算增益矩阵
    G = T.dot(T.dot(P,B.T), T.nlinalg.matrix_inverse(T.dot(T.dot(B,P),B.T)+Qv)) 

    # 计算新的状态
    update_W_vec = W_vec  +  T.dot(a,G.T) #(T.dot(G, a)).T

    # 计算新的状态协方差阵
    delta_P = -T.dot(T.dot(G,B), P) + Qw 
    update_P = [(P, P + delta_P)] 

    # 逆矢量化
    bi = 0
    delta_W = []
    for i in numpy.arange(len(W)):
        be = bi+W[i].size
        delta_tmp = update_W_vec[bi:be]
        delta_W.append( delta_tmp.reshape(W[i].shape) )
        bi = be

    update_W = [ (_W, _dW) for (_W, _dW) in  zip(W, delta_W) ]

    update_W.extend(update_P)

    update_Qw = [(Qw,  1.0 * Qw)]

    update_W.extend(update_Qw)

    f_train = theano.function([x, y], T.dot(a, a.T), updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode,
                                    givens=[(H, h_init)],
                                    on_unused_input='warn')

    return f_train, P
    
# 设置网络参数
n_input = 7
n_hidden = 10
n_output = 1
n_epochs = 5

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data('mackey_glass')

print data_x.shape, data_y.shape

print 'network: n_in:{},n_hidden:{},n_out:{}'.format(n_input, n_hidden, n_output)

# 构造网络
x_in = T.vector()   # 输入向量,第1维是时间
y_out = T.vector()  # 输出向量
lr = T.scalar()     # 学习速率，标量

H = T.matrix()      # 隐单元的初始化值
    

h_init = theano.shared(numpy.zeros((1,n_hidden), dtype=dtype), name='h_init') # 网络隐层初始值

W_in = [theano.shared(numpy.random.uniform(size=(1, n_hidden), low= -0.01, high=0.01).astype(dtype), 
                      name='W_in' + str(u)) for u in range(n_input)]                
b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")

W_hid = theano.shared(numpy.random.uniform(size=(n_hidden, n_hidden), low= -0.01, high=0.01).astype(dtype), name='W_hid') 

W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),low=-0.01,high=0.01).astype(dtype),name="W_out")
b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")

params = []
params.extend(W_in)
params.extend([b_in])
params.extend([W_hid])

input_taps = range(1-n_input, 1)
output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=dict(input=x_in, taps=input_taps),  # 从输出值中延时-1抽取
                        outputs_info=dict(initial = H, taps=output_taps),
                        non_sequences=params)
params.extend([W_out])
params.extend([b_out])                        
y,updates = theano.scan(purelin,
                        sequences=h_tmp,
                        non_sequences=params)
y = T.flatten(y)

batch_size = 3    # 设置的足够大时，等价于GD

print 'Batch Size: ', batch_size 

f_train,P = extend_kalman_train(params, y, batch_size, y_out, x_in)
                                                    
sim_fn = theano.function([x_in], outputs=y, givens=[(H, h_init)])

    
start_time = time.clock()     

for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(data_y.shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = data_y[train_index] 
        _x, _y = data_get_data_x_y(sub_seq, n_input)
        train_err = f_train(_x, _y)
        print '{}.{}: cost={}'.format(epochs_index, batch_index, train_err) 
y_sim = sim_fn(data_x[:-1])  
print 'y_sim.shape: ', y_sim.shape

plt.plot(range(data_y.shape[0]), data_y,'k')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim, 'r')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim - data_y[n_input:], 'g')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

plt.show() 
       
print "finished!"