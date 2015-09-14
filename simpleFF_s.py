# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:22:26 2015
使用前向网络进行频率识别：直接延时法

@author: simon
"""
import sys,time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def step(*args):
    #global n_input, n_hidden 
    x =  [args[u] for u in xrange(n_input)]    
    W_in =  [args[u] for u in xrange(n_input, n_input * 2)]
    b_in = args[n_input * 2]
    
    
    h = T.dot(x[0], W_in[0]) + b_in
    for j in xrange(1, n_input):
        h = h +  T.dot(x[j], W_in[j]) + b_in
        
    return T.tanh(h)
    
def purelin(*args):
    h = args[0]
    W_in =  [args[u] for u in xrange(1, n_input + 1)]
    b_in = args[n_input + 1]
    W_in
    b_in
    W_out = args[n_input + 2]
    b_out = args[n_input + 3]

    y = T.dot(h,W_out) + b_out
    return T.tanh(y)
    
# 设置网络参数
learning_rate = 0.001  
n_input = 4
n_hidden = 10
n_output = 1
N = 400
n_epochs = 2000

dtype=theano.config.floatX

# 加要处理的数据
data = numpy.genfromtxt("mytestdata.txt")
sampleNum = 400-n_input
index = range(sampleNum)
data_x = data[:,0]
data_y = data[:,1]

print data_x.shape, data_y.shape


print 'network: n_in:{},n_hidden:{},n_out:{}'.format(n_input, n_hidden, n_output)

# 构造网络
x_in = T.vector()   # 输入向量,第1维是时间
y_out = T.vector()  # 输出向量
lr = T.scalar()     # 学习速率，标量

h_init = numpy.zeros((n_hidden,n_hidden )).astype(dtype) # 网络隐层状态

W_in = [theano.shared(numpy.random.uniform(size=(1, n_hidden), low= -.01, high=.01).astype(dtype), 
                   name='W_in' + str(u)) for u in range(n_input)]                
b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),low=-0.01,high=0.01).astype(dtype),name="W_out")
b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")

params = []
params.extend(W_in)
params.extend([b_in])

h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=dict(input=x_in, taps = range(1-n_input, 1)),  # 从输出值中延时-1抽取
                        non_sequences=params)

params.extend([W_out])
params.extend([b_out])                        
y,updates = theano.scan(purelin,
                        sequences=h_tmp,
                        non_sequences=params)
y = T.flatten(y)                                                                                           
cost = ((y_out[n_input-1:,]-y)**2).sum()


# 编译表达式

gparams = []
for param in params:
    gparam = T.grad(cost, param)
    gparams.append(gparam)

# specify how to update the parameters of the model as a dictionary
updates = []
for param, gparam in zip(params, gparams):
    updates.append((param, param - lr * gparam))
    
# define the train function
train_fn = theano.function([x_in, y_out],                             
                     outputs=cost,
                     updates=updates,
                     givens=[(lr,T.cast(learning_rate, 'floatX'))])
                                         
sim_fn = theano.function([x_in],                             
                     outputs=y,
                     givens=[(lr,T.cast(learning_rate, 'floatX'))])

print 'Running ({} epochs)'.format(n_epochs)        
start_time = time.clock()     

for epochs_index in xrange(n_epochs) :             
    print train_fn(data_x, data_y)
    print 'Training {}'.format(epochs_index) 
  
y_sim = sim_fn(data_x)  
print y_sim.shape
print b_in.get_value() 

plt.plot(range(y_sim.shape[0]), y_sim, 'r')
plt.plot(range(data_x.shape[0]), data_x,'b')
plt.plot(range(data_y.shape[0]), data_y,'k')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

# 绘图

         
print "finished!"