# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
import time
import sys
import matplotlib.pyplot as plt
import copy

import utilities.datagenerator as DG
reload(DG)

compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
# SEED = int(numpy.random.lognormal()*100)
SEED = 123
numpy.random.seed(SEED)

def lstm_layer(n_input, n_LSTM, x):
    '''
    i f o c 统一处理
    '''
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(*args):
        '''
        x_ : 延时输入的x
        h_ : 前一时刻单元的输出
        c_ : 前一时刻单元的Cell值
        '''
        
        x =  [args[u] for u in xrange(n_input)]
        h_ = args[n_input]
        c_ = args[n_input+1]

        preact = T.dot(x[0], W_in[0])
        for i in xrange(1,n_input):
            preact += T.dot(x[i], W_in[i])

        preact += T.dot(h_, W_hid)  #  h的后向
        preact += b_in   

        i = T.nnet.sigmoid(_slice(preact, 0, n_LSTM)) # input gate
        f = T.nnet.sigmoid(_slice(preact, 1, n_LSTM)) # forget gate
        o = T.nnet.sigmoid(_slice(preact, 2, n_LSTM)) # output gate
        c = T.tanh(_slice(preact, 3, n_LSTM))         # cell state pre   

        c = f * c_ + i * c                                        # cell state

        h = o * T.tanh(c)                                         # unit output

        return h, c
    
    out_h = theano.shared(numpy.zeros((1,n_LSTM), dtype=theano.config.floatX), name="out_h")
    out_c = theano.shared(numpy.zeros((1,n_LSTM), dtype=theano.config.floatX), name="out_c")
    
    input_taps = range(1-n_input, 1)
    rval, updates = theano.scan(_step,
                                sequences=dict(input=x,taps=input_taps),
                                outputs_info=[out_h, out_c])
    return rval[0]   # 对外只有h

build_method = 5  # 0: LSTM
init_method = 0   # 0: normal   1: uniform

# 设置网络参数
n_input = 7
n_hidden = 15
n_output = 1
n_epochs = 5

saveto = 'MaskRNN_b{}_i{}_h{}_nh{}_S{}.npz'.format(
          build_method, init_method, n_hidden, 0, SEED) 
print 'Result will be saved to: ', saveto

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

x = T.vector()      # 输入向量,第1维是时间
y = T.vector()  # 输出向量, 第1维是时间


W_in = [theano.shared(numpy.random.uniform(size=(1, 4*n_hidden), low= -0.01, high=0.01).astype(dtype), 
                        name='W_in' + str(u)) for u in range(n_input)]
b_in_value = numpy.zeros((4 * n_hidden,), dtype=dtype)
b_in_value[1*n_hidden : 2*n_hidden] = 5  # large forget gate                
b_in = theano.shared(numpy.zeros((4 * n_hidden,), dtype=dtype), name="b_in")

W_hid = theano.shared(numpy.random.uniform(size=(n_hidden, 4*n_hidden), low= -0.01, high=0.01).astype(dtype), name='W_hid') 

W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),low=-0.01,high=0.01).astype(dtype),name="W_out")
b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")
params = []
params.extend(W_in)
params.extend([b_in])
params.extend([W_hid])
params.extend([W_out])   
params.extend([b_out])  

h_tmp = lstm_layer(n_input, n_hidden, x)  
  
pred = T.dot(h_tmp, W_out) + b_out

pred = theano.tensor.flatten(pred)

f_pred = theano.function([x], pred, name='f_pred')

cost = ((pred - y)**2).sum()

batch_size = 2    # 设置的足够大时，等价于GD

print 'Batch Size: ', batch_size 


update_W, P, cost = DG.PublicFunction.extend_kalman_train(params, pred, batch_size, y)

f_train = theano.function([x, y], cost, updates=update_W,
                                name='EKF_f_train',
                                mode=compile_mode)

sim_fn = theano.function([x],outputs=pred)  
start_time = time.clock() 

for epochs_index in xrange(n_epochs) :  

    kf = DG.DataPrepare.get_seq_minibatches_idx(train_data.shape[0], batch_size, n_input, shuffle=False)

    for batch_index, train_index in kf:
        sub_seq = train_data[train_index] 
        _x, _y = DG.PublicFunction.data_get_data_x_y(sub_seq, n_input)
        train_err = f_train(_x, _y)
        print '{}.{}: cost={}'.format(epochs_index, batch_index, train_err)

x_train_end = copy.deepcopy(train_data[-n_input:]) 

n_predict = 150
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

numpy.savez(saveto, cumulative_error=cumulative_error_list)

plt.show() 
       
print "finished!"
