# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:22:26 2015
使用Elman网络（简单局部回归网络）
以分块矩阵的形式组织网络

@author: simon
"""
import sys,time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from collections import OrderedDict

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def step(*args):
    #global n_input, n_hidden
    print args
    current_index = 0; 
    x =  [args[u] for u in xrange(n_input)] 
    current_index += n_input

    x_mask = args[current_index]
    current_index += 1

    hid_taps = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h  
    
    W_in =  [args[u] for u in xrange(current_index, current_index + n_segment_x * n_segment_h)]
    current_index += n_segment_x * n_segment_h

    b_in = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h

    W_hid = [args[u] for u in xrange(current_index, current_index + n_segment_h ** 2)]

    # 构造符合要求的分块的x
    x_perblock = n_input/n_segment_x
    x_block = []
    for i in xrange(n_segment_x):
        x_tmp = []
        for j in xrange(x_perblock):
            x_tmp.extend([x[i*x_perblock + j]])
        theano.tensor.stack(x_tmp)    
        x_block.extend([x_tmp])
    
    # 前向部分
    h_list = []

    for i in xrange(n_segment_h):
        h_tmp = T.dot(W_in[i*n_segment_x + 0], x_block[0])
        for j in xrange(1,n_segment_x):
            h_tmp += T.dot(W_in[i*n_segment_x + j], x_block[j])
        h_tmp = h_tmp + b_in[i]
        h_list.extend([h_tmp])

    # 回归部分
    h_list_r = []
    for i in xrange(n_segment_h):
        h_tmp = T.dot(W_hid[i*n_segment_h + 0],hid_taps[0])
        for j in xrange(i,n_segment_h):  # 下标从i开始，可以使其成为上三角矩阵
            h_tmp += T.dot(W_hid[i*n_segment_h + j],hid_taps[j])
        h_list_r.extend([h_tmp])
        h_list[i] += h_list_r[i]   # sum

    for i in xrange(n_segment_h):
        h_list[i] = theano.tensor.switch(x_mask[i],h_list[i], hid_taps[i])

    return [T.tanh(h_list[i]) for i in xrange(n_segment_h)]
    
def purelin(*args):
    print args
    current_index = 0
    h = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h

    W_out = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h

    b_out = [args[u] for u in xrange(current_index, current_index +n_segment_h)]
    current_index += n_segment_h

    print W_out, b_out

    y = T.dot(W_out[0], h[0]) + b_out[0]
    for j in xrange(1,n_segment_h):
        y += T.dot(W_out[j],h[j]) + b_out[j]

    return T.tanh(y)

def adadelta(lr, tparams, grads, x, x_mask, y, cost):
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

    f_grad_shared = theano.function([x, x_mask, y], cost, updates=updates_1,
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
n_input = 15
n_hidden = 15
n_segment_h = 5   # 隐层单元进行分块的块数，需保证整除
n_segment_x = 1   # 输入进行分块的块数，需保证整除
n_output = 1
N = 400
n_epochs = 500

dtype=theano.config.floatX

theano.config.exception_verbosity = 'high'

# 加要处理的数据
data = numpy.genfromtxt("mytestdata.txt")
sampleNum = 400-n_input
index = range(sampleNum)
data_x = data[:,0]
data_y = data[:,1]

data_x = numpy.zeros_like(data_x)
data_y = 1.0 * numpy.sin(30 * numpy.pi * numpy.linspace(0,1,400))

data_mask = numpy.zeros((sampleNum,n_segment_h), dtype=numpy.bool)

for t in xrange(sampleNum):
    for e in xrange(n_segment_h):
        if t % 2**e == 0:
            data_mask[t,e] = 1

print 'data_mask:', data_mask.shape
print data_mask

# 构造网络
x_in = T.vector()   # 输入向量,第1维是时间
x_mask = T.matrix()
y_out = T.vector()  # 输出向量
    

h_init = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
          name='h_init'+ str(u)) for u in range(n_segment_h)] 

# 生成系数矩阵
mu,sigma = 0.0, 0.1
numpy.random.normal(loc=mu, scale=sigma, size=(n_hidden/n_segment_h, n_input/n_segment_x))
W_in = [theano.shared(numpy.random.normal(loc=mu, scale=sigma, size=(n_hidden/n_segment_h, n_input/n_segment_x)).astype(dtype), 
                      name='W_in' + str(u)) for u in range(n_segment_h * n_segment_x)]                
b_in = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
                      name="b_in" + str(u)) for u in range(n_segment_h)]
W_hid = [theano.shared(numpy.random.normal(size=(n_hidden/n_segment_h, n_hidden/n_segment_h), 
                        loc=mu, scale=sigma).astype(dtype), 
                        name='W_hid'+ str(u)) for u in range(n_segment_h * n_segment_h)] 
W_out = [theano.shared(numpy.random.normal(size=(n_output,n_hidden/n_segment_h),
                       loc=mu, scale=sigma).astype(dtype),
                       name="W_out"+ str(u)) for u in range(n_segment_h)]
b_out = [theano.shared(numpy.zeros((n_output,), dtype=dtype),
                       name="b_out"+ str(u)) for u in range(n_segment_h)]

params = []
params.extend(W_in)
params.extend(b_in)
params.extend(W_hid)

input_taps = range(1-n_input, 1)
output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=[dict(input=x_in, taps=input_taps), x_mask],  # 从输出值中延时-1抽取
                        outputs_info=h_init,   # taps = [-1], default
                        non_sequences=params)

params.extend(W_out)   
params.extend(b_out)  

params_1 = []
params_1.extend(W_out)   
params_1.extend(b_out)                        
y,updates = theano.scan(purelin,
                        sequences=h_tmp,
                        non_sequences=params_1)
y = T.flatten(y)                    
                        
cost = ((y_out[n_input:,]-y)**2).sum()

params4grad = []

params4grad.extend(W_in)
params4grad.extend(b_in)
#params4grad.extend(W_hid)   #左下部分不参与计算
for i in xrange(n_segment_h):
    for j in xrange(i,n_segment_h):
        params4grad.extend([W_hid[i*n_segment_h+j]])

params4grad.extend(W_out)   
params4grad.extend(b_out)

# 编译表达式
grads = theano.tensor.grad(cost, wrt=params4grad)
tparams = OrderedDict()
for p in params4grad:
    tparams[p.name] = p   

lr_v = 0.0001
lr_ada = theano.tensor.scalar(name='lr_ada')

f_pred = theano.function([x_in, x_mask],                             
                         outputs=y)  

updates_1, updates_2, f_grad_shared, f_update = adadelta(lr_ada, tparams, grads, x_in, x_mask, y_out, cost)

print 'data info:', data_x.shape, data_y.shape
start_time = time.clock()   
for epochs_index in xrange(n_epochs) :  
        print 'cost = {}: {}'.format(epochs_index, f_grad_shared(data_x, data_mask, data_y))
        f_update(lr_v)
    
y_sim = f_pred(data_x, data_mask) 

print y_sim.shape

plt.plot(range(data_y.shape[0]-y_sim.shape[0],data_y.shape[0]), y_sim, 'r')
plt.plot(range(data_x.shape[0]), data_x,'b')
plt.plot(range(data_y.shape[0]), data_y,'k')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))
        
print "finished!"

plt.show()

