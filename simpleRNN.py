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

def step(x,h, W_in, W_r, W_out):
    total = T.dot(W_in,x) + T.dot(W_r,h)
    
    return T.tanh(total)
    
def purelin(h, W_out):
    y = T.dot(h,W_out)
    return y
    
# 设置网络参数
learning_rate = 0.01             
n_input = 10
n_hidden = 10
n_output = 1
N = 1000
n_epochs = 10

dtype=theano.config.floatX

# 加要处理的数据
data = numpy.genfromtxt("mytestdata.txt")

#y = data[:,0]
#y = y.T
#t = data[:,1]
#t = t.T

print 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_input, n_hidden, n_output)

# 构造网络
x_in = T.matrix()   # 输入向量,第1维是时间
y_out = T.vector()  # 输出向量
H = T.matrix()      # 隐层单元状态,用来输入初始值
lr = T.scalar()     # 学习速率，标量

h_init = numpy.zeros((n_hidden,n_hidden )).astype(dtype) # 网络隐层状态

W_in  = theano.shared(numpy.random.uniform(size=(n_input,n_hidden),low=-0.01,high=0.01).astype(dtype),name="W_in")
W_r   = theano.shared(numpy.random.uniform(size=(n_hidden,n_hidden),low=-0.01,high=0.01).astype(dtype),name="W_r")
W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),low=-0.01,high=0.01).astype(dtype),name="W_out")

params = []
params.extend([W_in])
params.extend([W_r])
params.extend([W_out])

h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=x_in,
                        outputs_info=dict(initial=H, taps=[-1]),  # 从输出值中延时-1抽取
                        non_sequences=params)
                                               
y = T.sum(T.dot(h_tmp, W_out),0)
                                                                                            
#cost = ((y_out - y.reshape(y_out.shape[0])) ** 2).sum()  
                         
#cost =  1/2 * T.sqrt( T.sum(( T.pow(y_out,2) - T.pow(y,2))))

#cost = -T.mean(T.log(y)[T.arange(y_out.shape[0]), 0]) 

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
    
print updates

# define the train function
learning_rate = 0.01    
train_fn = theano.function([x_in, y_out],                             
                     outputs=cost,
                     updates=updates,
                     givens=[(H,T.cast(h_init, 'floatX')),   #设置网络初值 
                            (lr,T.cast(learning_rate, 'floatX'))])

# 数据输入,训练
data_x = numpy.random.uniform(size=(N,n_input)).astype(theano.config.floatX)
data_y = numpy.random.uniform(size=(N,)).astype(theano.config.floatX)

print 'Running ({} epochs)'.format(n_epochs)        
start_time = time.clock()     
for epochs_index in xrange(n_epochs) :             
    print train_fn(data_x, data_y)
    print 'Training {}'.format(epochs_index)   
                          
print >> sys.stderr, ('overall time (%.5fs)' % ((time.clock() - start_time) / 1.))

# 绘图

         
print "finished!"