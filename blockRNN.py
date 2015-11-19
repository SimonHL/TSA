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

import utilities.datagenerator as DG
reload(DG)

def step(*args):
    #global n_input, n_hidden
    print args
    current_index = 0; 
    x =  [args[u] for u in xrange(n_input)] 
    current_index += n_input

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
        for j in xrange(i,n_segment_h):
            h_tmp += T.dot(W_hid[i*n_segment_h + j],hid_taps[j])

        h_list_r.extend([h_tmp])
        h_list[i] += h_list_r[i]   # sum

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
  
    
# 设置网络参数
n_input = 6
n_hidden = 10
n_segment_h = 2   # 隐层单元进行分块的块数，需保证整除
n_segment_x = 2   # 输入进行分块的块数，需保证整除
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
data_y = numpy.sin(8 * numpy.pi * numpy.linspace(0,1,400))

# 构造网络
x_in = T.vector()   # 输入向量,第1维是时间
y_out = T.vector()  # 输出向量
    

h_init = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
          name='h_init'+ str(u)) for u in range(n_segment_h)] 

# 生成系数矩阵
W_in = [theano.shared(numpy.random.uniform(size=(n_hidden/n_segment_h, n_input/n_segment_x), 
                      low= -0.01, high=0.01).astype(dtype), 
                      name='W_in' + str(u)) for u in range(n_segment_h * n_segment_x)]                
b_in = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
                      name="b_in" + str(u)) for u in range(n_segment_h)]
W_hid = [theano.shared(numpy.random.uniform(size=(n_hidden/n_segment_h, n_hidden/n_segment_h), 
                        low= -0.01, high=0.01).astype(dtype), 
                        name='W_hid'+ str(u)) for u in range(n_segment_h * n_segment_h)] 
W_out = [theano.shared(numpy.random.uniform(size=(n_output,n_hidden/n_segment_h),
                       low=-0.01,high=0.01).astype(dtype),
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
                        sequences=dict(input=x_in, taps=input_taps),  # 从输出值中延时-1抽取
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
                        
cost = ((y_out[n_input-1:,]-y)**2).sum()

params4grad = []

params4grad.extend(W_in)
params4grad.extend(b_in)
#params4grad.extend(W_hid)   #左下部分不参与计算
for i in xrange(n_segment_h):
    for j in xrange(i,n_segment_h):
        params4grad.extend([W_hid[i*n_segment_h+j]])

params.extend(W_out)   
params.extend(b_out)  


# 编译表达式
grads = theano.tensor.grad(cost, wrt=params)
tparams = OrderedDict()
for p in params:
    tparams[p.name] = p   

lr_v = 0.0001
lr_ada = theano.tensor.scalar(name='lr_ada')

f_pred = theano.function([x_in],                             
                         outputs=y)  

updates_1, updates_2, f_grad_shared, f_update = DG.PublicFunction.adadelta(lr_ada, tparams, grads, [x_in, y_out], cost)

start_time = time.clock()   
for epochs_index in xrange(n_epochs) :  
        print 'cost = {}: {}'.format(epochs_index, f_grad_shared(data_x, data_y))
        f_update(lr_v)
    
y_sim = f_pred(data_x) 

print y_sim.shape

plt.plot(range(y_sim.shape[0]), y_sim, 'r')
plt.plot(range(data_x.shape[0]), data_x,'b')
plt.plot(range(data_y.shape[0]), data_y,'k')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))
        
print "finished!"

plt.show()

