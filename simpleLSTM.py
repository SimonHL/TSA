# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
import time
import sys
import matplotlib.pyplot as plt

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)
    
def ortho_weight(ndim):
    '''
    做奇异值分解，返回特征向量矩阵U
    '''
    W = 0.1 * numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    """
    初始化LSTM的参数
    W 是LSTM对应几个Gate的输入系数，对应分别为：input, forget, output, 
    U 是LSTM的回归系数
    b 是偏置系数
    """
    
    # 对第1维进行连接得到    (dim, 4*dim)
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))  # (4*dim)
    b[options['dim_proj']: 2 * options['dim_proj']] = 5
    params[_p(prefix, 'b')] = b.astype(theano.config.floatX)

    return params

def init_params(options):
    """
    程序用到的全局变量，以有序字典的方式存放在params中
    Wemb 是
    """
    params = OrderedDict()
    
    # LSTM层的系数
    params = param_init_lstm(options,
                              params,
                              prefix=options['encoder'])
    # 输出层的系数
    params['U'] = 0.1 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(theano.config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(theano.config.floatX)

    return params
    
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
def lstm_layer(tparams, x_sequence, options, prefix='lstm'):
    '''
    
    '''
    
    nsteps = x_sequence.shape[0]

    # (n_t, 4*dim)
    state_below = (T.dot(x_sequence, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])   

    dim_proj = options['dim_proj']
    
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(x_, h_, c_):
        '''
        x_ : 单元的输入数据: W x + b
        h_ : 前一时刻单元的输出
        c_ : 前一时刻单元的Cell值
        '''
                
        
        preact = T.dot(h_, tparams[_p(prefix, 'U')])  # (4*dim)
        preact += x_    # h 延时后加权的目标维数 和 Wx+b的维数相同，都是LSTM单元的个数的4倍，可以直接相加

        i = T.nnet.sigmoid(_slice(preact, 0, dim_proj)) # input gate
        f = T.nnet.sigmoid(_slice(preact, 1, dim_proj)) # forget gate
        o = T.nnet.sigmoid(_slice(preact, 2, dim_proj)) # output gate
        c = T.tanh(_slice(preact, 3, dim_proj))         # cell state pre   

        c = f * c_ + i * c                                        # cell state

        h = o * T.tanh(c)                                         # unit output

        return h, c
    
    out_h = theano.shared(numpy.zeros((1,dim_proj), dtype=theano.config.floatX), name="out_h")
    out_c = theano.shared(numpy.zeros((1,dim_proj), dtype=theano.config.floatX), name="out_c")

    rval, updates = theano.scan(_step,
                                sequences=state_below,
                                outputs_info=[out_h, out_c],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]
    
def build_model(tparams, options):

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    # 以下变量的第1维是：时间
    x = T.matrix()
    y = T.vector()

    proj = lstm_layer(tparams, x, options,
                        prefix=options['encoder'])
        
    proj = theano.tensor.reshape(proj, (proj.shape[0], proj.shape[2]))
      
   # pred = T.tanh(T.dot(proj, tparams['U']) + tparams['b'])
    
    pred = T.dot(proj, tparams['U']) + tparams['b']
    
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
#
#    off = 1e-8
#    if pred.dtype == 'float16':
#        off = 1e-6
    
    pred = theano.tensor.flatten(pred)
    
    cost = ((pred - y)**2).sum()

    return use_noise, x, y, f_pred_prob, cost


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

    f_grad_shared = theano.function([x, y], cost, updates=updates_1,
                                    name='adadelta_f_grad_shared',
                                    mode='FAST_COMPILE')

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
                               mode='FAST_COMPILE')

    return updates_1, updates_2,f_grad_shared, f_update

def train_lstm(
    dim_proj=40,  # 输入x的个数和LSTM单元个数相等 
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=1500,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):
    


    # Model options
    model_options = locals().copy()
    
    # 加要处理的数据
    data = numpy.genfromtxt("mytestdata.txt")
    n_input = dim_proj
    sampleNum = 400-n_input
    index = range(sampleNum)
    data_x = numpy.zeros((sampleNum,n_input))
    data_y = numpy.zeros((sampleNum,))

    for i in index:
        #data_x[i,:] = data[i:i + n_input , 0]
        data_y[i] = data[i + n_input, 1]
    
    data_y = numpy.sin(8 * numpy.pi * numpy.linspace(0,1,sampleNum))
    ydim = 1

    model_options['ydim'] = ydim
    
   
    params = init_params(model_options)
    
    print "params:", params.keys()
    
    tparams = init_tparams(params)
    
    
    print 'Building model... '    
    # use_noise is for dropout
    (use_noise, x, y, f_pred, cost) = build_model(tparams, model_options)
    
   
    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = theano.tensor.grad(cost, wrt=tparams.values())  
    
    f_grad = theano.function([x, y], grads, name='f_grad')
    

    lr = theano.tensor.scalar(name='lr')
    updates_1, updates_2, f_grad_shared, f_update = optimizer(lr, tparams, grads, x, y, cost)

    start_time = time.time()

    for epochs_index in xrange(max_epochs) :  
        print 'cost = {}: {}'.format(epochs_index, f_grad_shared(data_x, data_y))
        f_update(lrate)
    # print 'Training {}'.format(epochs_index) 
    
#    for k, p in tparams.iteritems():
#                print '%s:' %k,  p.get_value()[0] 

    
    y_sim = f_pred(data_x) 

    end_time = time.time()

    print >> sys.stderr, ('Training took %.1fs' %
                      (end_time - start_time)) 
    
    plt.plot(range(y_sim.shape[0]), y_sim, 'r')
#    plt.plot(range(data_x.shape[0]), data_x,'b')
#    plt.plot(range(data_y.shape[0]), data_y,'k')

    plt.show()
    
    
    print "end"
    

if __name__ == '__main__':
    train_lstm(
        test_size=500,
    )


