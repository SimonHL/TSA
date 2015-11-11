# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:22:26 2015
使用Elman网络（简单局部回归网络）

load phoneme
p = con2seq(y);
t = con2seq(t);
lrn_net = newlrn(p,t,8);
lrn_net.trainFcn = 'trainbr';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 50;
lrn_net = train(lrn_net,p,t);

y = sim(lrn_net,p);
plot(cell2mat(y));


@author: simon
"""
import sys,time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import utilities.datagenerator as DG


compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def step(*args):
    #global n_input, n_hidden 
    print args
    x =  [args[u] for u in xrange(n_input)] 
    hid_taps = args[n_input]  
    
    W_in =  [args[u] for u in xrange(n_input + 1, n_input * 2 + 1)]
    b_in = args[n_input * 2 + 1]
    W_hid = args[n_input * 2 + 2]
    
    
    h = T.dot(x[0], W_in[0]) + b_in
    for j in xrange(1, n_input):              # 前向部分
        h = h +  T.dot(x[j], W_in[j]) + b_in
    
    h = h + T.dot(hid_taps, W_hid)            # 回归部分

        
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
    return T.tanh(y)

def extend_kalman_train(W, y_hat, y, x):
    '''
    P: 状态对应的协方差矩阵
    Qv: 观测噪声的协方差矩阵
    Qw: 输入噪声的协方差矩阵
    W: 需要估计的状态
    y_hat: 系统的输出
    y: 期望的输出

    注意：P, Qv, Qw, W的元素个数相同，类型均为list, 对应不同的W
    '''

    # 系数矩阵的向量化
    W_vec = [_W.flatten() for _W in W]


    # 初始值
    P = [theano.shared( numpy.eye(p.get_value().size ) * numpy_floatX(100.0)  ) for p in W]

    Qv = theano.shared( numpy.eye(sampleNum) * numpy_floatX(0.0001) )  # 观测噪声协方差矩阵

    # 输入噪声协方差矩阵， 
    Qw = [theano.shared( numpy.eye(p.get_value().size ) * numpy_floatX(10.0)  ) for p in W] 

    # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数
    B = []
    for _W in W:
        J, updates = theano.scan(lambda i, y_hat, W: T.grad(y_hat[i], _W).flatten(), 
                                 sequences=T.arange(y_hat.shape[0]), 
                                 non_sequences=[y_hat, _W])
        B.extend([J])

    # 计算残差
    a = y[n_input-1:] - y_hat

    # 计算增益矩阵
    G = [T.dot(T.dot(_P,_B.T), T.nlinalg.matrix_inverse(T.dot(T.dot(_B,_P),_B.T)+Qv)) for (_P,_B) in zip(P,B)]

    # 计算新的状态
    update_W_vec = [_W  +  T.dot(_G, a) for (_G, _W) in zip(G,W_vec)] 

    # 计算新的状态协方差阵
    delta_P = [ -T.dot(T.dot(_G,_B), _P) + _Qw for (_G, _P, _B, _Qw) in zip(G, P,B,Qw) ]
    update_P = [ (_P, _dP) for (_P, _dP) in zip(P,delta_P) ]

    # 逆矢量化
    delta_W = [_W_vec.reshape(_W.shape ) for (_W, _W_vec) in  zip(W, update_W_vec) ]
    update_W = [ (_W, _dW) for (_W, _dW) in  zip(W, delta_W) ]

    update_W.extend(update_P)

    update_Qw = [(_Qw, _Qw * 0.9) for _Qw in Qw]

    update_W.extend(update_Qw)

    f_train = theano.function([x, y], T.dot(a, a.T), updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode,
                                    givens=[(H, h_init)],
                                    on_unused_input='warn')

    return f_train

def get_minibatches_idx(n, minibatch_size, overhead, shuffle=False):
    """
    对总的时间序列进行切片
    n : 序列的总长度 
    minibatch_size : 切片的序列长度
    overhead : x 映射到y 时的延时长度
    shuffle : 是否进行重排。 对于时间序列，原始数据不能重排，
    """

    idx_list = numpy.arange(n, dtype="int32")

    minibatches = []
    minibatch_start = 0
    end_index = n - overhead
    for i in range(end_index // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size + overhead])
        minibatch_start += minibatch_size

    # if (minibatch_start != end_index):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    if shuffle:
        numpy.random.shuffle(minibatches)

    return zip(range(len(minibatches)), minibatches)
    
# 设置网络参数
learning_rate = 0.001
n_input = 4
n_hidden = 10
n_output = 1
n_epochs = 10

dtype=theano.config.floatX

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data(0)

print data_x.shape, data_y.shape

print 'network: n_in:{},n_hidden:{},n_out:{}'.format(n_input, n_hidden, n_output)

# 构造网络
x_in = T.vector()   # 输入向量
y_out = T.vector()  # 输出向量
lr = T.scalar()     # 学习速率，标量

H = T.matrix()      # 隐单元的初始化值
    
mu = 0
sigma = 0.1
h_init = theano.shared(numpy.zeros((1,n_hidden), dtype=dtype), name='h_init') # 网络隐层初始值

W_in = [theano.shared(numpy.random.normal(size=(1, n_hidden), loc=mu, scale=sigma).astype(dtype), 
                      name='W_in' + str(u)) for u in range(n_input)]                
b_in = theano.shared(numpy.random.normal(size=(n_hidden,), loc=mu, scale=sigma).astype(dtype), name="b_in")
W_hid = theano.shared(numpy.random.normal(size=(n_hidden, n_hidden), loc=mu, scale=sigma).astype(dtype), name='W_hid') 
W_out = theano.shared(numpy.random.normal(size=(n_hidden, n_output), loc=mu, scale=sigma).astype(dtype), name="W_out")
b_out = theano.shared(numpy.random.normal(size=(n_output,), loc=mu, scale=sigma).astype(dtype), name="b_out")

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


overhead = n_input - 1 
batch_size = data_x.shape[0] - n_input + 1   #设置的足够大时，等价于GD


kf = get_minibatches_idx(data_x.shape[0], batch_size, overhead, shuffle=True)

sampleNum = kf[0][1].shape[0]-n_input+1

print 'sampleNum: ', sampleNum 

f_train = extend_kalman_train(params, y, y_out, x_in)
                                                    
sim_fn = theano.function([x_in],                             
                    outputs=y,
                    givens=[(H, h_init)])

print 'Running ({} epochs)'.format(n_epochs)        
start_time = time.clock()     

for epochs_index in xrange(n_epochs) :  

    kf = get_minibatches_idx(data_x.shape[0], batch_size, overhead, shuffle=False)

    for batch_index, train_index in kf:
        _x = [data_x[t] for t in train_index ]
        _y = [data_y[t] for t in train_index ]

        train_err = f_train(_x, _y)

        print '{}.{}: cost={}'.format(epochs_index, batch_index, train_err) 

 
y_sim = sim_fn(data_x)  
print y_sim.shape
print b_in.get_value() 

plt.plot(range(data_x.shape[0]), data_x,'b')
plt.plot(range(data_y.shape[0]), data_y,'k')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim, 'r')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

plt.show()         
print "finished!"