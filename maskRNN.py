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
import copy

import utilities.datagenerator as DG
reload(DG)

compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'
build_method = 1 # 0: block  1: CW-RNN 2: MaskRNN
init_method = 0   # 0: normal   1: uniform

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)

def step(*args):
    #global n_input, n_hidden
    print 'step args: ', args
    current_index = 0; 
    x =  [args[u] for u in xrange(n_input)] 
    current_index += n_input

    x_mask = args[current_index]
    current_index += 1

    hid_taps = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h  
    
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
        if build_method == 2:
            h_tmp = T.dot(W_in[i*n_segment_x + i], x_block[i]) # 下标从i开始，可以使其成为上三角矩阵
            for j in xrange(i+1,n_segment_x):
                h_tmp += T.dot(W_in[i*n_segment_x + j], x_block[j])
        else:
            h_tmp = T.dot(W_in[i*n_segment_x + 0], x_block[0])
            for j in xrange(1,n_segment_x):
                h_tmp += T.dot(W_in[i*n_segment_x + j], x_block[j])
        h_tmp = h_tmp + b_in[i]
        h_list.extend([h_tmp])

    # 回归部分
    h_list_r = []
    for i in xrange(n_segment_h):
        if build_method == 0:
            h_tmp = T.dot(W_hid[i*n_segment_h + 0],hid_taps[0])
            for j in xrange(1,n_segment_h):
                h_tmp += T.dot(W_hid[i*n_segment_h + j],hid_taps[j])
        else:   # 方式1 和方式2 都需要三角化
            h_tmp = T.dot(W_hid[i*n_segment_h + i],hid_taps[i]) # 下标从i开始，可以使其成为上三角矩阵
            for j in xrange(i+1,n_segment_h):  
                h_tmp += T.dot(W_hid[i*n_segment_h + j],hid_taps[j])
 
        h_list_r.extend([h_tmp])
        h_list[i] += h_list_r[i]   # sum

    for i in xrange(n_segment_h):
        h_list[i] = theano.tensor.switch(x_mask[i],h_list[i], hid_taps[i])
        #h_list[i] = x_mask[i] * h_list[i] + (1-x_mask[i]) * hid_taps[i]

    return [T.tanh(h_list[i]) for i in xrange(n_segment_h)]
    
def purelin(*args):
    h = [args[u] for u in xrange(n_segment_h)]
   
    y = T.dot(W_out[0], h[0]) + b_out[0]
    for j in xrange(1,n_segment_h):
        y += T.dot(W_out[j],h[j]) + b_out[j]

    return y # T.tanh(y)

def  gen_random_mask(sampleNum,n_segment_h):
    data_mask = numpy.zeros((sampleNum,n_segment_h), dtype=numpy.bool)

    random_e = numpy.random.exponential(scale=0.5, size=(sampleNum,))

    for t in xrange(sampleNum):
    #     i = numpy.floor(random_e[t])
    #     if i >= n_segment_h:
    #         i = n_segment_h-1

    #     data_mask[t,0:i] = 1

        for e in xrange(n_segment_h):
            if t % 2**e == 0:
                data_mask[t,e] = 1
        
    return data_mask

def prepare_data(data_x, data_mask, data_y):
    '''
    将数据分为训练集，验证集和测试集
    '''
    data_len = len(data_y)
    train_end = numpy.floor(data_len * 0.5)
    valid_end = numpy.floor(data_len * 0.8)

    train_data_x = data_x[:train_end]
    train_data_mask = data_mask[:train_end,:]
    train_data_y = data_y[:train_end]

    valid_data_x = data_x[train_end:valid_end]
    valid_data_mask = data_mask[train_end:valid_end,:]
    valid_data_y = data_y[train_end:valid_end]

    test_data_x = data_x[valid_end:data_len]
    test_data_mask = data_mask[valid_end:data_len,:]
    test_data_y = data_y[valid_end:data_len]

    train_data = [train_data_x, train_data_mask, train_data_y]
    valid_data = [valid_data_x, valid_data_mask, valid_data_y]
    test_data = [test_data_x, test_data_mask, test_data_y]

    return train_data, valid_data, test_data 

'''
主程序
'''
# 设置网络参数
n_input = 7      # 输入数据的长度
n_hidden = 15
n_segment_h = 3   # 隐层单元进行分块的块数，需保证整除
n_segment_x = 1   # 输入进行分块的块数，需保证整除， MaskRNN需要保证能够形成对角矩阵
n_output = 1

n_epochs = 20

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data('mackey_glass')
N = data_y.shape[0]

# sampleNum = 400-n_input

data_mask = gen_random_mask(N,n_segment_h)
print 'data_mask:', data_mask.shape
print data_mask

train_data, valid_data, test_data = prepare_data(data_x, data_mask, data_y)

###########################################################
# 构造网络
x = T.vector()      # 输入向量,第1维是时间
x_mask = T.matrix()
t = T.vector()      # 输出向量
    

h_init = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
          name='h_init'+ str(u)) for u in range(n_segment_h)] 

# 生成系数矩阵
mu,sigma = 0.0, 0.1
# numpy.random.normal(loc=mu, scale=sigma, size=(n_hidden/n_segment_h, n_input/n_segment_x))
if init_method == 0:
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
else:
    W_in = [theano.shared(numpy.random.uniform(size=(n_hidden/n_segment_h, n_input/n_segment_x), 
                          low= -0.01, high=0.01).astype(dtype), 
                          name='W_in' + str(u)) for u in range(n_segment_h * n_segment_x)]                
    b_in = [theano.shared(numpy.zeros((n_hidden/n_segment_h,), dtype=dtype), 
                          name="b_in" + str(u)) for u in range(n_segment_h)]
    W_hid = [theano.shared(numpy.random.uniform(size=(n_hidden/n_segment_h, n_hidden/n_segment_h), 
                           low= -0.01, high=0.01).astype(dtype), 
                           name='W_hid'+ str(u)) for u in range(n_segment_h * n_segment_h)] 
    W_out = [theano.shared(numpy.random.uniform(size=(n_output,n_hidden/n_segment_h),
                           low= -0.01, high=0.01).astype(dtype), 
                           name="W_out"+ str(u)) for u in range(n_segment_h)]
    b_out = [theano.shared(numpy.zeros((n_output,), dtype=dtype),
                           name="b_out"+ str(u)) for u in range(n_segment_h)]

params = []
params.extend(W_in)
params.extend(b_in)
params.extend(W_hid)

start_compile_time = time.clock()  

input_taps = range(1-n_input, 1)
output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                        sequences=[dict(input=x, taps=input_taps), x_mask],  # 从输出值中延时-1抽取
                        outputs_info=h_init,   # taps = [-1], default
                        non_sequences=params)

params.extend(W_out)   
params.extend(b_out)  
                     
y,updates = theano.scan(purelin, sequences=h_tmp)
y = T.flatten(y)                    
                        
cost = ((t-y)**2).sum()

params4grad = []


if build_method == 2:
    params4grad.extend(W_in)
else:   #前向部分，左下部分不参与计算
    for i in xrange(n_segment_h):
        for j in xrange(i,n_segment_x):
            params4grad.extend([W_in[i*n_segment_h+j]])

params4grad.extend(b_in)

if build_method == 0:
    params4grad.extend(W_hid)
else:   #回归部分，左下部分不参与计算
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

sim_fn = theano.function([x, x_mask], outputs=y)
pred_cost = theano.function([x, x_mask, t], outputs=cost)  

updates_1, updates_2, f_grad_shared, f_update = DG.PublicFunction.adadelta(lr_ada, tparams, grads, [x, x_mask, t], cost)

######################################
# train
print 'train info:', train_data[0].shape, train_data[1].shape, train_data[2].shape
print 'valid info:', valid_data[0].shape, valid_data[1].shape, valid_data[2].shape
print 'test info:', test_data[0].shape, test_data[1].shape, test_data[2].shape
history_errs = numpy.zeros((n_epochs,3), dtype=dtype)  
history_errs_cur_index = 0
patience = 10
valid_fre = 1
bad_counter = 0

batch_size = 400   #设置的足够大时，等价于GD

start_time = time.clock()   
for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(train_data[2].shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = train_data[2][train_index] 
        _mask = copy.deepcopy(train_data[1][train_index])
        _x, _y = DG.PublicFunction.data_get_data_x_y(sub_seq, n_input)
        _mask = _mask[:-1]
        train_err = f_grad_shared(_x, _mask, _y)
        f_update(lr_v)
        print '{}.{}: train error={:.3f}'.format(epochs_index, batch_index, float(train_err))

    if numpy.mod(epochs_index+1, valid_fre) == 0:    
        valid_err = pred_cost(valid_data[0][:-1], valid_data[1][:-1], valid_data[2][n_input:])
        test_err = pred_cost(test_data[0][:-1], test_data[1][:-1], test_data[2][n_input:])

        print '{}: train error={:.3f}, valid error={:.3f}, test error={:.3f}'.format(
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

# 使用Valid Data测试多步预测误差
x_train_end = copy.deepcopy(train_data[0][-n_input:]) 
x_train_mask_end = train_data[1][-1]

n_predict = 100
y_predict = numpy.zeros((n_predict,))
cumulative_error = 0
cumulative_error_list = numpy.zeros((n_predict,))
for i in numpy.arange(n_predict):
    x_train_mask_end.resize((1,x_train_mask_end.shape[0]))
    y_predict[i] = sim_fn(x_train_end,x_train_mask_end)
    x_train_end[:-1] = x_train_end[1:]
    x_train_end[-1] = y_predict[i]
    x_train_mask_end = valid_data[1][i]
    cumulative_error += numpy.abs(y_predict[i] - valid_data[2][i])
    cumulative_error_list[i] = cumulative_error
plt.figure(3)
plt.plot(numpy.arange(n_predict), cumulative_error_list)
plt.title('cumulative error')
plt.grid(True)

plt.figure(4)
plt.plot(numpy.arange(y_predict.shape[0]), y_predict,'r')
plt.plot(numpy.arange(valid_data[2].shape[0]), valid_data[2],'g')


y_sim = sim_fn(data_x[:-1], data_mask[:-1])
y_sim_index = numpy.arange(n_input, data_x.shape[0]) 
print y_sim.shape

plt.figure(1)
index_start = data_x.shape[0]-y_sim.shape[0]
index_train_end = train_data[0].shape[0]
index_valid_end = index_train_end + valid_data[0].shape[0]
index_test_end = index_valid_end + test_data[0].shape[0]
plt.plot( range(index_start, index_train_end),     y_sim[:index_train_end-index_start], 'r')
plt.plot( range(index_train_end, index_valid_end), y_sim[index_train_end-index_start:index_valid_end-index_start], 'g')
plt.plot( range(index_valid_end, index_test_end),  y_sim[index_valid_end-index_start:index_test_end-index_start], 'b')

plt.plot( range(data_y.shape[0]), data_y,'k')

plt.figure(2)
plt.plot( range(history_errs_cur_index),  history_errs[:history_errs_cur_index,0], 'r')
plt.plot( range(history_errs_cur_index),  history_errs[:history_errs_cur_index,1], 'g')
plt.plot( range(history_errs_cur_index),  history_errs[:history_errs_cur_index,2], 'b')
plt.show()
                          
print 'compile time (%.5fs), run time (%.5fs)' % ((time.clock() - start_compile_time) / 1., (time.clock() - start_time) / 1.)
        
print "finished!"


