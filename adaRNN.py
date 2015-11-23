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
import copy

import utilities.datagenerator as DG
reload(DG)

compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)

def step(*args):
    #global n_input, n_hidden 
    print args
    x =  [args[u] for u in xrange(n_input)] 
    hid_taps = args[n_input]    
    
    h = T.dot(x[0], W_in[0])
    for j in xrange(1, n_input):           # 前向部分
        h +=  T.dot(x[j], W_in[j])
    
    h += T.dot(hid_taps, W_hid)            # 回归部分
    h += b_in                              # 偏置部分
  
    return T.tanh(h)
    
def purelin(*args):
    print args
    h = args[0]

    y = T.dot(h,W_out) + b_out
    return y #T.tanh(y)
    
# 设置网络参数
n_input = 7
n_hidden = 15
n_output = 1
n_epochs = 20

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data('mackey_glass')

index_test_begin = data_y.shape[0] / 2
train_data_index = numpy.arange(index_test_begin)
test_data_index = numpy.arange(index_test_begin, data_y.shape[0])
train_data = data_y[train_data_index]
test_data = data_y[test_data_index]

print 'train_data.shape: ', train_data.shape
print 'test_data.shape: ', test_data.shape

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
params.extend([W_out])   
params.extend([b_out])   

input_taps = range(1-n_input, 1)
output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=dict(input=x_in, taps=input_taps),  # 从输出值中延时-1抽取
                        outputs_info=h_init)
                     
y,updates = theano.scan(purelin,  sequences=h_tmp)
y = T.flatten(y)                    
                        
cost = ((y_out-y)**2).sum()

batch_size = 200    # 设置的足够大时，等价于GD

print 'Batch Size: ', batch_size 
grads = theano.tensor.grad(cost, wrt=params)
tparams = OrderedDict()
for p in params:
    tparams[p.name] = p   

lr_v = 0.0001
lr_ada = theano.tensor.scalar(name='lr_ada')

updates_1, updates_2, f_grad_shared, f_update = DG.PublicFunction.adadelta(lr_ada, tparams, grads, [x_in, y_out], cost)

sim_fn = theano.function([x_in],outputs=y)  
start_time = time.clock() 

for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(train_data.shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = train_data[train_index] 
        _x, _y = DG.PublicFunction.data_get_data_x_y(sub_seq, n_input)
        train_err = f_grad_shared(_x, _y)
        f_update(lr_v)
        print '{}.{}: cost={}'.format(epochs_index, batch_index, train_err)

x_train_end = copy.deepcopy(train_data[-n_input:]) 

n_predict = 100
y_predict = numpy.zeros((n_predict,))
cumulative_error = 0
cumulative_error_list = numpy.zeros((n_predict,))
for i in numpy.arange(n_predict):
    y_predict[i] = sim_fn(x_train_end)
    x_train_end[:-1] = x_train_end[1:]
    x_train_end[-1] = y_predict[i]
    cumulative_error += numpy.abs(y_predict[i] - test_data[i])
    cumulative_error_list[i] = cumulative_error
plt.figure(3)
plt.plot(numpy.arange(n_predict), cumulative_error_list)
plt.title('cumulative error')
plt.grid(True)

plt.figure(1)
plt.plot(numpy.arange(y_predict.shape[0]), y_predict,'r')
plt.plot(numpy.arange(300), test_data[:300],'g')

y_sim = sim_fn(data_x[:-1])  # 整体的单步误差
print 'y_sim.shape: ', y_sim.shape

plt.figure(2)
plt.plot(range(data_y.shape[0]), data_y,'k')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim, 'r')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim - data_y[n_input:], 'g')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

plt.show() 
       
print "finished!"