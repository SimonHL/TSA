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

compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

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
        h_tmp = T.dot(W_in[i*n_segment_x + i], x_block[i]) # 下标从i开始，可以使其成为上三角矩阵
        for j in xrange(i+1,n_segment_x):
            h_tmp += T.dot(W_in[i*n_segment_x + j], x_block[j])
        h_tmp = h_tmp + b_in[i]
        h_list.extend([h_tmp])

    # 回归部分
    h_list_r = []
    for i in xrange(n_segment_h):
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
    print 'purelin args: ', args
    current_index = 0
    h = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h

    W_out = [args[u] for u in xrange(current_index, current_index + n_segment_h)]
    current_index += n_segment_h

    b_out = [args[u] for u in xrange(current_index, current_index +n_segment_h)]
    current_index += n_segment_h

    print 'purelin Check W_out, b_out: ', W_out, b_out

    y = T.dot(W_out[0], h[0]) + b_out[0]
    for j in xrange(1,n_segment_h):
        y += T.dot(W_out[j],h[j]) + b_out[j]

    return T.tanh(y)

def extend_kalman_train(W, y_hat, dim_y_hat, y, x):
    '''
    P: 状态对应的协方差矩阵
    Qv: 观测噪声的协方差矩阵
    Qw: 输入噪声的协方差矩阵
    W: 需要估计的状态
    y_hat: 系统的输出
    dim_y_hat: y_hat 的维数
    y: 期望的输出

    注意：P, Qv, Qw, W的元素个数相同，类型均为list, 对应不同的W
    '''

    # 系数矩阵的向量化
    dim_Wv = 0
    W_vec = []
    for i in numpy.arange(len(W)):
        dim_Wv += W[i].get_value().size
        W_vec.extend([W[i].flatten()])
    W_vec = tuple(W_vec)
    W_vec = T.concatenate(W_vec)  

    print 'number of parameters: ',dim_Wv

    P = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0)  ) # 状态的协方差矩阵
    
    Qw = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0) )  # 输入噪声协方差矩阵， 
    
    Qv = theano.shared( numpy.eye(dim_y_hat) * numpy_floatX(0.01) )  # 观测噪声协方差矩阵

    # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数
    B = []
    for _W in W:
        J, updates = theano.scan(lambda i, y_hat, W: T.grad(y_hat[i], _W).flatten(), 
                                 sequences=T.arange(y_hat.shape[0]), 
                                 non_sequences=[y_hat, _W])
        B.extend([J])

    B = T.concatenate(tuple(B),axis=1)

    # 计算残差
    a = y - y_hat # 单步预测误差

    # 计算增益矩阵
    G = T.dot(T.dot(P,B.T), T.nlinalg.matrix_inverse(T.dot(T.dot(B,P),B.T)+Qv)) 

    # 计算新的状态
    update_W_vec = W_vec  +  T.dot(a,G.T) #(T.dot(G, a)).T

    # 计算新的状态协方差阵
    delta_P = -T.dot(T.dot(G,B), P) + Qw 
    update_P = [(P, P + delta_P)] 

    # 逆矢量化
    bi = 0
    delta_W = []
    for i in numpy.arange(len(W)):
        be = bi+W[i].size
        delta_tmp = update_W_vec[bi:be]
        delta_W.append( delta_tmp.reshape(W[i].shape) )
        bi = be

    update_W = [ (_W, _dW) for (_W, _dW) in  zip(W, delta_W) ]

    update_W.extend(update_P)

    update_Qw = [(Qw,  1.0 * Qw)]

    update_W.extend(update_Qw)

    f_train = theano.function([x, y], T.dot(a, a.T), updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode,
                                    givens=[(H, h_init)],
                                    on_unused_input='warn')

    return f_train, P

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
                                    mode=compile_mode)

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
                               mode=compile_mode)

    return updates_1, updates_2,f_grad_shared, f_update    

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
    train_end = numpy.floor(data_len * 0.6)
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

def get_minibatches_idx(n, minibatch_size, overhead, shuffle=False):
    """
    对总的时间序列进行切片
    n : 序列的总长度 
    minibatch_size : 切片的序列长度
    overhead : x 映射到y 时的延时长度
    shuffle : 是否进行重排。 对于时间序列，原始数据不能重排，
    """

    idx_list = numpy.arange(n, dtype="int32")

    minibatches = []
    minibatch_start = 0
    end_index = n - overhead
    for i in range(end_index // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size + overhead])
        minibatch_start += minibatch_size

    if (minibatch_start != end_index):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    if shuffle:
        numpy.random.shuffle(minibatches)

    return zip(range(len(minibatches)), minibatches)

def extend_kalman_train(W, y_hat, y, x, x_mask):
    '''
    P: 状态对应的协方差矩阵
    Qv: 观测噪声的协方差矩阵
    Qw: 输入噪声的协方差矩阵
    W: 需要估计的状态
    y_hat: 系统的输出
    y: 期望的输出

    注意：P, Qv, Qw, W的元素个数相同，类型均为list, 对应不同的W
    '''

    # 初始值
    P = [theano.shared( numpy.eye(p.get_value().shape[0])*numpy_floatX(1.0)  ) for p in W]
    Qv = [theano.shared( numpy.eye(p.get_value().shape[0])*numpy_floatX(1.0)  ) for p in W]
    Qw = [theano.shared( numpy.eye(p.get_value().shape[0])*numpy_floatX(1.0)  ) for p in W]


    # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数
    B = []
    for _W in W:
        J, updates = theano.scan(lambda i, y_hat,W: T.grad(y_hat[i], _W), 
                                 sequences=T.arange(y_hat.shape[0]), 
                                 non_sequences=[y_hat, _W])
        B.extend([J])

    # 计算残差
    a = y - y_hat

    # 计算增益矩阵
    G = [_P * _B.T * (_B * _P * _B.T + _Qv) for (_P,_B,_Qv) in zip(P,B,Qv)]

    # 计算新的状态
    update_W = [(_W, _W + _G * a.T ) for (_G, _W) in zip(G,W)]

    # 计算新的状态协方差阵
    update_P = [ (_P, _P - _G * _B * _P + _Qw) for (_P, _B, _Qw) in zip(P,B,Qw) ]

    update_W.extend(update_P)


    f_train = theano.function([x, x_mask, y], y_hat, updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode)

    return f_train

   

'''
主程序
'''
# 设置网络参数
n_input = 15      # 输入数据的长度
n_hidden = 15
n_segment_h = 5   # 隐层单元进行分块的块数，需保证整除
n_segment_x = 5   # 输入进行分块的块数，需保证整除， MaskRNN需要保证能够形成对角矩阵
n_output = 1

n_epochs = 1

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

# 加要处理的数据
g = DG.Generator()
data_x,data_y = g.get_data('mackey_glass')

data_mask = gen_random_mask(N,n_segment_h)
print 'data_mask:', data_mask.shape
print data_mask

train_data, valid_data, test_data = prepare_data(data_x, data_mask, data_y)

###########################################################
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

start_compile_time = time.clock()  

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
                        
cost = ((y_out[n_input-1:]-y)**2).sum()

params4grad = []

# params4grad.extend(W_in)
for i in xrange(n_segment_h):
    for j in xrange(i,n_segment_x):
        params4grad.extend([W_in[i*n_segment_h+j]])
params4grad.extend(b_in)
#params4grad.extend(W_hid)   #左下部分不参与计算
for i in xrange(n_segment_h):
    for j in xrange(i,n_segment_h):
        params4grad.extend([W_hid[i*n_segment_h+j]])

params4grad.extend(W_out)   
params4grad.extend(b_out)

f_train =  extend_kalman_train(params4grad, y, y_out, x_in, x_mask)

# # 编译表达式
# grads = theano.tensor.grad(cost, wrt=params4grad)
# tparams = OrderedDict()
# for p in params4grad:
#     tparams[p.name] = p   

# lr_v = 0.0001
# lr_ada = theano.tensor.scalar(name='lr_ada')

# f_pred = theano.function([x_in, x_mask], outputs=y)
# pred_cost = theano.function([x_in, x_mask, y_out], outputs=cost)  

# updates_1, updates_2, f_grad_shared, f_update = adadelta(lr_ada, tparams, grads, x_in, x_mask, y_out, cost)


# ######################################
# # train
# print 'train info:', train_data[0].shape, train_data[1].shape, train_data[2].shape
# print 'valid info:', valid_data[0].shape, valid_data[1].shape, valid_data[2].shape
# print 'test info:', test_data[0].shape, test_data[1].shape, test_data[2].shape
# history_errs = numpy.zeros((n_epochs,3), dtype=dtype)  
# history_errs_cur_index = 0
# patience = n_epochs
# valid_fre = 1
# bad_counter = 0

# overhead = n_input - 1 
# batch_size = 20   #设置的足够大时，等价于GD

# start_time = time.clock()   
# for epochs_index in xrange(n_epochs) :  

#     kf = get_minibatches_idx(len(train_data[0]), batch_size, overhead, shuffle=True)

#     for batch_index, train_index in kf:
#         _x = [train_data[0][t]for t in train_index]
#         _mask = [train_data[1][t]for t in train_index]
#         _y = [train_data[2][t]for t in train_index]


#         train_err = f_grad_shared(_x, _mask, _y)
#         f_update(lr_v)

#         #print '{}:{} train_batch error={:.3f}'.format(epochs_index, batch_index, float(train_err))

#     if numpy.mod(epochs_index+1, valid_fre) == 0:    
#         valid_err = pred_cost(valid_data[0], valid_data[1], valid_data[2])
#         test_err = pred_cost(test_data[0], test_data[1], test_data[2])

#         print '{}: train error={:.3f}, valid error={:.3f}, test error={:.3f}'.format(
#             epochs_index, float(train_err), float(valid_err), float(test_err))

#         history_errs[history_errs_cur_index,:] = [train_err, valid_err, test_err]
#         history_errs_cur_index += 1

#         if valid_err <= history_errs[:history_errs_cur_index,1].min():
#             bad_counter = 0

#         if history_errs_cur_index > patience and  valid_err >= history_errs[:history_errs_cur_index-patience,1].min():
#             bad_counter += 1
#             if bad_counter > patience:
#                 print 'Early Stop!'
#                 break
  

# y_sim = f_pred(data_x, data_mask) 

# print y_sim.shape

# index_start = data_x.shape[0]-y_sim.shape[0]
# index_train_end = train_data[0].shape[0]
# index_valid_end = index_train_end + valid_data[0].shape[0]
# index_test_end = index_valid_end + test_data[0].shape[0]

# plt.figure(1)

# plt.plot( range(index_start, index_train_end),     y_sim[:index_train_end-index_start], 'r')
# plt.plot( range(index_train_end, index_valid_end), y_sim[index_train_end-index_start:index_valid_end-index_start], 'g')
# plt.plot( range(index_valid_end, index_test_end),  y_sim[index_valid_end-index_start:index_test_end-index_start], 'b')

# plt.plot( range(data_x.shape[0]), data_x,'b.')
# plt.plot( range(data_y.shape[0]), data_y,'k')

# plt.figure(2)
# plt.plot( range(history_errs_cur_index),  history_errs[:,0], 'r')
# plt.plot( range(history_errs_cur_index),  history_errs[:,1], 'g')
# plt.plot( range(history_errs_cur_index),  history_errs[:,2], 'b')
# plt.show()
                          
# print 'compile time (%.5fs), run time (%.5fs)' % ((time.clock() - start_compile_time) / 1., (time.clock() - start_time) / 1.)
        
print "finished!"


