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
    return  y #T.tanh(y)

def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
                 
    updates_1 = zgup + rg2up

    f_grad_shared = theano.function([x, y], cost, updates=updates_1,
                                    name='adadelta_f_grad_shared',
                                    mode='FAST_COMPILE')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    
    updates_2 = ru2up + param_up;

    f_update = theano.function([lr], [], updates=updates_2,
                               on_unused_input='ignore',
                               name='adadelta_f_update',
                               mode='FAST_COMPILE')

    return updates_1, updates_2,f_grad_shared, f_update    
    
# 设置网络参数
n_input = 7
n_hidden = 10
n_output = 1
n_epochs = 10

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
                        outputs_info=h_init,
                        non_sequences=params)
params.extend([W_out])   
params.extend([b_out])                        
y,updates = theano.scan(purelin,
                        sequences=h_tmp,
                        non_sequences=params)
y = T.flatten(y)                    
                        
                        
cost = ((y_out-y)**2).sum()

batch_size = 100    # 设置的足够大时，等价于GD

print 'Batch Size: ', batch_size 
grads = theano.tensor.grad(cost, wrt=params)
tparams = OrderedDict()
for p in params:
    tparams[p.name] = p   

lr_v = 0.0001
lr_ada = theano.tensor.scalar(name='lr_ada')

sim_fn = theano.function([x_in],outputs=y)  

updates_1, updates_2, f_grad_shared, f_update = adadelta(lr_ada, tparams, grads, x_in, y_out, cost)

start_time = time.clock() 

for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(data_y.shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = data_y[train_index] 
        _x, _y = data_get_data_x_y(sub_seq, n_input)
        train_err = f_grad_shared(_x, _y)
        f_update(lr_v)
        print '{}.{}: cost={}'.format(epochs_index, batch_index, train_err)

y_sim = sim_fn(data_x[:-1])  
print 'y_sim.shape: ', y_sim.shape

plt.plot(range(data_y.shape[0]), data_y,'k')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim, 'r')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim - data_y[n_input:], 'g')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

plt.show() 
       
print "finished!"