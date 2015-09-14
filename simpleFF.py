# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:22:26 2015
使用前向网络进行频率识别：数据构造法
@author: simon
"""
import sys,time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def step(x, W_in, b_in):
    total = T.dot(x,W_in) + b_in
    return T.tanh(total)
    
    #return T.tanh(total)
    
def purelin(h, W_out, b_out):
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
data_x = numpy.zeros((sampleNum,n_input))
data_y = numpy.zeros((sampleNum,1))
for i in index:
    data_x[i,:] = data[i:i + n_input , 0]
    data_y[i,:] = data[i + n_input, 1]

print data_x.shape, data_y.shape
# 数据输入
#data_x = numpy.random.uniform(size=(N,n_input)).astype(theano.config.floatX)
#data_y = numpy.random.uniform(size=(N,1)).astype(theano.config.floatX)



print 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_input, n_hidden, n_output)

# 构造网络
x_in = T.matrix()   # 输入向量,第1维是时间
y_out = T.matrix()  # 输出向量
lr = T.scalar()     # 学习速率，标量

h_init = numpy.zeros((n_hidden,n_hidden )).astype(dtype) # 网络隐层状态

W_in  = theano.shared(numpy.random.uniform(size=(n_input,n_hidden),low=-0.01,high=0.01).astype(dtype),name="W_in")
b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),low=-0.01,high=0.01).astype(dtype),name="W_out")
b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")

params = []
params.extend([W_in])
params.extend([b_in])
params.extend([W_out])
params.extend([b_out])

h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=x_in,  # 从输出值中延时-1抽取
                        non_sequences=[W_in,b_in])
                        
y,updates = theano.scan(purelin,
                        sequences=h_tmp,
                        non_sequences=[W_out,b_out])
                                                                                            
cost = ((y_out-y)**2).sum()


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
plt.plot(range(data_x.shape[0]), data_x[:,n_input-1],'b')
plt.plot(range(data_y.shape[0]), data_y[:,0],'k')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

# 绘图

         
print "finished!"