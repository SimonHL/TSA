# -*- coding: utf-8 -*-
'''
Pipelined RNN
'''

import numpy
import theano
import theano.tensor as T 
from collections import OrderedDict
import matplotlib.pyplot as plt

dtype=theano.config.floatX
theano.config.exception_verbosity = 'low'
compile_mode = 'FAST_RUN'
#compile_mode = 'FAST_COMPILE'

def build_rnn_last(x, y, N, module_index,W_in, b, W_hid):
    '''
    一个正常的RNN
    x：输入向量，第1维是时间，第2维是RNN单元的输入维数
    N：每个RNN模块中隐单元的数目
    '''

    input_taps = [-module_index]
    print input_taps

    def _step(_x, _h, _WI, _b, _WR):
        h = T.dot(_x, _WI) + _b 
        h += T.dot(_h, _WR)            # 回归部分

        return T.tanh(h)

    h_init = theano.shared(numpy.zeros((N,), dtype=dtype), name='h_init_first') # 网络隐层初始值
    h_tmp, updates = theano.scan(_step,  # 计算BPTT的函数
                                sequences=dict(input=x, taps=input_taps),  # 从输出值中延时-1抽取
                                outputs_info=h_init,
                                non_sequences=[W_in, b, W_hid])


    y_rnn = h_tmp[:,0]                # 第一个隐单元的输出
    e_rnn = y_rnn - y[module_index:]

    print 'build_rnn_last OK'
    return  y_rnn, e_rnn               

def build_rnn(x, y, y_first, N, M,module_index, W_in, b, W_hid):
    '''
    中间级联用的RNN，使用了前一时刻的输出
    '''

    def _step(_x, _y_first, _h, _WI, _b, _WR):
        h = T.dot(_x, _WI) + _b           # 输入部分
        h += _y_first * _WR[0,:]          # 级联部分
        h += T.dot(_h[1:], _WR[1:,:])     # 非级联回归部分

        return T.tanh(h)
    
    input_taps = [-module_index]
    print input_taps
    h_init = theano.shared(numpy.zeros((N,), dtype=dtype),) # 网络隐层初始值
    h_tmp, updates = theano.scan(_step,  # 计算BPTT的函数
                                sequences=[dict(input=x, taps=input_taps), y_first],  # 从输出值中延时-1抽取
                                outputs_info=h_init,
                                non_sequences=[W_in, b, W_hid])

    y_rnn = h_tmp[:,0]                # 第一个隐单元的输出
    e_rnn = y_rnn - y[M:]

    print 'build_rnn OK'
    return  y_rnn, e_rnn 

def data_gen_x(n):
    if n < 250:
        return numpy.sin(numpy.pi * n / 25.0)
    elif n < 500:
        return 1.0
    elif n < 750:
        return -1.0 
    elif n <= 1000:
        tmp = 0.3*numpy.sin(numpy.pi * n / 25)
        tmp += 0.1*numpy.sin(numpy.pi * n / 32)
        tmp += 0.6*numpy.sin(numpy.pi * n / 10)
        return tmp
    else:
        return 0.0

def data_dynamic_gen_1():
    data_x = numpy.zeros((1000,))
    data_y = numpy.zeros((1000,),)
    for i in xrange(1000):
        data_x[i] = data_gen_x(i)
        if i > 2:
            tmp =  0.72 * data_y[i-1]
            tmp += 0.025*data_y[i-2] * data_x[i-1] 
            tmp += 0.001 * data_x[i-2]**2 
            tmp += 0.2 * data_x[i-3]
            data_y[i] = tmp

    return data_x, data_y

def data_dynamic_gen_2():
    data_x = numpy.zeros((1000,))
    data_y = numpy.zeros((1000,),)
    for i in xrange(1000):
        data_x[i] = data_gen_x(i)
        if i > 2:
            tmp =  data_y[i-1] * data_y[i-2] * data_y[i-3] * data_x[i-1]
            tmp = tmp * (data_y[i-3] - 1) + data_x[i]
            tmp = tmp / (1 + data_y[i-2]**2 + data_y[i-3]**2)
            data_y[i] = tmp

    return data_x, data_y
    
def load_data():
    '''
    原始数据的读取
    '''
    data_x, data_y = data_dynamic_gen_1()
    return data_x,data_y

def prepare_data(data_x, data_y, p, M):
    '''
    将信号整理为需要的格式
    '''
    data_len = len(data_y)
    input_len = data_len - p + 1
    input_x = numpy.zeros((input_len, p))
    input_y = numpy.zeros((input_len,))
    for i in xrange(input_len):
        input_y[i] = data_y[i + p - 1]
        input_x[i] = data_x[i : i + p]
    
    return input_x, input_y
        

def build_prnn(x, y, N, M, W_in, b, W_hid):
    '''
    x：输入向量，第1维是时间，第2维是RNN单元的输入维数
    N：每个RNN模块中隐单元的数目

    '''
    y_pred = []
    e_pred = []

    module_index = M
    y_rnn, e_rnn = build_rnn_last(x, y, N, module_index,W_in, b, W_hid)
    y_pred.extend([y_rnn])
    e_pred.extend([e_rnn])

    for module_index in xrange(M-1,0,-1):
        y_rnn, e_rnn = build_rnn(x, y, y_rnn, N, M, module_index, W_in, b, W_hid)
        y_pred.extend([y_rnn])
        e_pred.extend([e_rnn])

    return y_pred, e_pred

def adadelta(lr, tparams, grads, x, y, cost):
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

    zipped_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=dtype),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=dtype),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=dtype),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
                 
    updates_1 = zgup + rg2up

    f_grad_shared = theano.function([x, y], cost, updates=updates_1,
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

    return updates_1, updates_2, f_grad_shared, f_update    

def prnn():
    '''
    main process
    '''

    p = 5  # dim of input vector
    M = 4  # module number
    N = 4  # hidden units per module
    
    data_x, data_y = load_data()

    input_x, input_y = prepare_data(data_x, data_y, p, M)

    x = T.matrix()   # 输入向量,第1维是时间
    y = T.vector()   # 输出向量
   
    # 网络参数
    W_in = theano.shared(numpy.random.uniform(size=(p, N), low= -0.01, high=0.01).astype(dtype), name='W_in')             
    b = theano.shared(numpy.zeros((N,), dtype=dtype), name="b")
    W_hid = theano.shared(numpy.random.uniform(size=(N, N), low= -0.01, high=0.01).astype(dtype), name='W_hid') 

    params = []
    params.extend([W_in])
    params.extend([b])
    params.extend([W_hid])

    y_pred, e_pred  = build_prnn(x, y, N, M, W_in, b, W_hid)

    # 损失函数
    cost = (e_pred[0] ** 2).sum() * (0.9 ** M) 
    for i  in xrange(1,M):
        cost += (e_pred[i] ** 2).sum() * (0.9 ** (M-1) )

    # 计算梯度
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # updates = []
    # lr = 0.002
    # for param, gparam in zip(params, gparams):
    #     updates.append((param, param - lr * gparam))
    lr_v = 0.0001
    lr_ada = theano.tensor.scalar(name='lr_ada')
    grads = theano.tensor.grad(cost, wrt=params)
    tparams = OrderedDict()
    for p in params:
        tparams[p.name] = p   
    updates_1, updates_2, f_grad_shared, f_update = adadelta(lr_ada, tparams, grads, x, y, cost)

    f_pred = theano.function([x], outputs=y_pred[M-1])  
    
    for epochs_index in xrange(1000) :             
        print 'Training {}: cost={}'.format(epochs_index, f_grad_shared(input_x, input_y))
        f_update(lr_v)
    
    print 'data info:', input_x.shape, input_y.shape

    y_sim = f_pred(input_x) 

    print 'y_sim.shape=', y_sim.shape
    plt.plot(range(input_x.shape[0]), input_x,'b')
    plt.plot(range(input_y.shape[0]), input_y,'k')
    plt.plot(range(input_y.shape[0]-y_sim.shape[0],input_y.shape[0]), y_sim, 'r')

    plt.show()



def test():
    data_x, data_y = data_dynamic_gen_1()
    data_x_2, data_y_2 = data_dynamic_gen_2()
    plt.plot(range(data_x.shape[0]), data_x,'k')
    plt.plot(range(data_y.shape[0]), data_y,'b')
    plt.plot(range(data_x_2.shape[0]), data_x_2 - data_x,'y')
    plt.plot(range(data_y_2.shape[0]), data_y_2,'r')

    plt.show()

if __name__ == '__main__':
    prnn()
    print 'OK'

