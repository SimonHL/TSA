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
    return T.tanh(y)
    
# 设置网络参数
learning_rate = 0.0005
n_input = 4
n_hidden = 10
n_output = 1
N = 400
n_epochs = 2000

dtype=theano.config.floatX

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data(0)

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
                     givens=[(lr,T.cast(learning_rate, 'floatX')),
                             (H, h_init)])                      
                             
                                         
sim_fn = theano.function([x_in],                             
                     outputs=y,
                     givens=[(lr,T.cast(learning_rate, 'floatX')),
                             (H, h_init)])

print 'Running ({} epochs)'.format(n_epochs)        
start_time = time.clock()     

for epochs_index in xrange(n_epochs):             
    print '{}: cost={}'.format(epochs_index, train_fn(data_x, data_y)) 
  
y_sim = sim_fn(data_x)  
print y_sim.shape

plt.plot(range(data_x.shape[0]), data_x,'b')
plt.plot(range(data_y.shape[0]), data_y,'k')
plt.plot(range(data_y.shape[0]-y_sim.shape[0], data_y.shape[0]), y_sim, 'r')
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

plt.show()   

         
print "finished!"