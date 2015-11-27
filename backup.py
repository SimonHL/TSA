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
SEED = int(numpy.random.lognormal()*100)
SEED = 999
numpy.random.seed(SEED)

def step(*args):
    #global n_input, n_hidden 
    print args
    x =  [args[u] for u in xrange(n_input)] 
    x_drive = args[n_input]
    hid_taps = args[n_input+1]    
    
    h = T.dot(x[0], W_in[0])
    for j in xrange(1, n_input):           # 前向部分
        h +=  T.dot(x[j], W_in[j])
    
    h += T.dot(hid_taps, W_hid)            # 回归部分
    h += b_in                              # 偏置部分

    g_update = T.nnet.sigmoid(b_ug + x_drive) # update gate

    h = g_update * h +  (1 - g_update) * hid_taps 
  
    return T.tanh(h)
    
def purelin(*args):
    print args
    h = args[0]

    y = T.dot(h,W_out) + b_out
    return y #T.tanh(y)

def  gen_drive_sin(sampleNum,N):
    '''
    生成一个长度为sampleNum, 周期为N的正弦信号
    '''
    data = 1.0 * numpy.sin(2 * numpy.pi / N  * numpy.arange(sampleNum))
    return data



def prepare_data(data_x, data_mask, data_y):
    '''
    将数据分为训练集，验证集和测试集

    注意，因为要进行hstack, 行向量会变为列向量
    '''
    data_len = len(data_y)
    train_end = numpy.floor(data_len * 0.5)
    test_end = numpy.floor(data_len * 0.8)

    if data_x.ndim == 1:
        data_x.resize((data_x.shape[0],1))
    if data_mask != []  and data_mask.ndim == 1:
        data_mask.resize((data_mask.shape[0],1))
    if data_y.ndim == 1:
        data_y.resize((data_y.shape[0],1))

    if data_mask == []:
        allData = numpy.concatenate((data_x,data_y), axis=1)
    else:
        allData = numpy.concatenate((data_x,data_mask,data_y), axis=1)

    train_data = allData[:train_end,...]
    test_data = allData[train_end:test_end,...]
    valid_data = allData[test_end:,...]

    return train_data, valid_data, test_data 

'''
主程序
'''
build_method = 5  # 0: RNN
init_method = 0   # 0: normal   1: uniform


# 设置网络参数
n_input = 7
n_hidden = 15
n_output = 1
n_epochs = 40

saveto = 'MaskRNN_b{}_i{}_h{}_nh{}_S{}.npz'.format(
          build_method, init_method, n_hidden, 0, SEED) 
print 'Result will be saved to: ', saveto

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data('mackey_glass')
drive_data = gen_drive_sin(data_y.shape[0], n_hidden)

train_data, valid_data, test_data = prepare_data(data_x, drive_data, data_y) # data_x 会成为列向量

print 'train_data.shape: ', train_data.shape
print 'test_data.shape: ', test_data.shape

# 构造网络
x_in = T.vector()   # 输入向量,第1维是时间
x_drive = T.vector() # 周期驱动信号
y_out = T.vector()  # 输出向量
lr = T.scalar()     # 学习速率，标量

H = T.matrix()      # 隐单元的初始化值

h_init = theano.shared(numpy.zeros((1,n_hidden), dtype=dtype), name='h_init') # 网络隐层初始值

mu,sigma = 0.0, 0.1
if init_method == 0:
    W_in = [theano.shared(numpy.random.normal(size=(1, n_hidden),
                          loc=mu, scale=sigma).astype(dtype), 
                          name='W_in' + str(u)) for u in range(n_input)]                
    b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
    W_hid = theano.shared(numpy.random.normal(size=(n_hidden, n_hidden), 
                          loc=mu, scale=sigma).astype(dtype), name='W_hid') 
    W_out = theano.shared(numpy.random.normal(size=(n_hidden,n_output),
                          loc=mu,scale=sigma).astype(dtype),name="W_out")
    b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")
else:
    W_in = [theano.shared(numpy.random.uniform(size=(1, n_hidden), 
                          low=-0.01, high=0.01).astype(dtype), 
                          name='W_in' + str(u)) for u in range(n_input)]                
    b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
    W_hid = theano.shared(numpy.random.uniform(size=(n_hidden, n_hidden), 
                          low=-0.01, high=0.01).astype(dtype), name='W_hid') 
    W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),
                          low=-0.01,high=0.01).astype(dtype),name="W_out")
    b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")
b_ug = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name='b_ug')
params = []
params.extend(W_in)
params.extend([b_in])
params.extend([W_hid])
params.extend([W_out])   
params.extend([b_out]) 
params.extend([b_ug])  

start_compile_time = time.clock()  
input_taps = range(1-n_input, 1)
output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=[dict(input=x_in, taps=input_taps), x_drive],  # 从输出值中延时-1抽取
                        outputs_info=dict(initial = H, taps=output_taps))
                       
y,updates = theano.scan(purelin, sequences=h_tmp)
y = T.flatten(y)

batch_size = 2    # 设置的足够大时，等价于GD

print 'Batch Size: ', batch_size 

update_W, P, cost = DG.PublicFunction.extend_kalman_train(params, y, batch_size, y_out)

f_train = theano.function([x_in, x_drive, y_out], cost, updates=update_W,
                                name='EKF_f_train',
                                mode=compile_mode,
                                givens=[(H, h_init)])                                              

sim_fn = theano.function([x_in, x_drive], outputs=y, givens=[(H, h_init)])
pred_cost = theano.function([x_in, x_drive, y_out], outputs=cost, givens=[(H, h_init)]) 
    
######################################
# train
print 'train info:', train_data.shape
print 'valid info:', valid_data.shape
print 'test info:', test_data.shape
history_errs = numpy.zeros((n_epochs,3), dtype=dtype)  
history_errs_cur_index = 0
patience = 100
valid_fre = 1
bad_counter = 0

start_time = time.clock()   
for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(train_data.shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = train_data[train_index,-1] 
        _x, _y = DG.PublicFunction.data_get_data_x_y(sub_seq, n_input)
        _d = train_data[train_index[n_input:],1]
        train_err = f_train(_x,_d,_y)
        print '{}.{}: train error={:.6f}'.format(epochs_index, batch_index, float(train_err))

    if numpy.mod(epochs_index+1, valid_fre) == 0: 
        test_err = pred_cost(test_data[:-1,0], test_data[n_input:,1], test_data[n_input:,-1])   
        valid_err = pred_cost(valid_data[:-1,0], valid_data[n_input:,1],valid_data[n_input:,-1]) 
        
        print '{}: train error={:.6f}, valid error={:.6f}, test error={:.6f}'.format(
            epochs_index, float(train_err), float(valid_err), float(test_err))

        history_errs[history_errs_cur_index,:] = [train_err, valid_err, test_err]
        history_errs_cur_index += 1

        if valid_err <= history_errs[:history_errs_cur_index,1].min():
            bad_counter = 0

        if history_errs_cur_index > patience and  valid_err >= history_errs[:history_errs_cur_index-patience,1].min():
            bad_counter += 1
            if bad_counter > patience:
                print 'Early Stop!'
                break
x_train_end = train_data[-n_input:,0]

n_predict = 150
y_predict = numpy.zeros((n_predict,))
cumulative_error = 0
cumulative_error_list = numpy.zeros((n_predict,))

for i in numpy.arange(n_predict):
    y_predict[i] = sim_fn(x_train_end, test_data[i:i+1,1])
    x_train_end[:-1] = x_train_end[1:]
    x_train_end[-1] = y_predict[i]
    cumulative_error += numpy.abs(y_predict[i] - test_data[i,-1])
    cumulative_error_list[i] = cumulative_error
plt.figure(3)
plt.plot(numpy.arange(n_predict), cumulative_error_list)
plt.title('cumulative error')
plt.grid(True)

plt.figure(4)
plt.plot(numpy.arange(y_predict.shape[0]), y_predict,'r')
plt.plot(numpy.arange(test_data.shape[0]), test_data[:,-1],'g')

y_sim = sim_fn(data_x[:-1,0], drive_data[n_input:,0])  # 整体的单步误差
print 'y_sim.shape: ', y_sim.shape

plt.figure(1)
index_start = data_x.shape[0]-y_sim.shape[0]
index_train_end = train_data.shape[0]
index_test_end = index_train_end + test_data.shape[0]
index_valid_end = index_test_end + valid_data.shape[0]
train_index = numpy.arange(index_train_end-index_start)
test_index  = numpy.arange(index_train_end-index_start,index_test_end-index_start)
valid_index = numpy.arange(index_test_end-index_start,index_valid_end-index_start)

plt.plot(train_index, y_sim[train_index],'r')
plt.plot(test_index, y_sim[test_index],'y')
plt.plot(valid_index, y_sim[valid_index],'b')
plt.plot(data_y,'k')  # 原始信号
plt.plot(y_sim-data_y[n_input:,0], 'g')

plt.figure(2)
plt.plot( history_errs[:history_errs_cur_index,0], 'r')
plt.plot( history_errs[:history_errs_cur_index,1], 'g')
plt.plot( history_errs[:history_errs_cur_index,2], 'b')

numpy.savez(saveto, cumulative_error=cumulative_error_list)

print b_ug.get_value()

# plt.show() 
                          
print 'compile time (%.5fs), run time (%.5fs)' % ((time.clock() - start_compile_time) / 1., (time.clock() - start_time) / 1.)
       
print "finished!"