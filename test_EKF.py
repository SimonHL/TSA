# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T


compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 100
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def get_seq_minibatches_idx(n, minibatch_size, overhead, shuffle=False):
    """
    对总的时间序列进行切片，从数据中随机选取一个起点
    Input：
    n : 序列的总长度 
    minibatch_size : 切片的序列长度，对于单输出而言，等价于BPTT的深度， 也等价于y的维数
    overhead: x 映射到y 时的延时长度，即x序列长度和输出y序列长度的差值
               对于单步预测，overhead = n_input
               对于滤波，overhead= n_input-1
    shuffle : 是否进行重排，用以选择随机的开始位置
    Return：
    minibatches_index： 子序列的编号
    minibatches： 子序列
    """

    idx_list = numpy.arange(n)

    len_sub_seq = overhead+minibatch_size       # 子序列的长度
    num_sub_seq = n-len_sub_seq+1               # 子序列的个数 
    start_list = numpy.arange(num_sub_seq)      # 起始位置列表
    if shuffle:
        numpy.random.shuffle(start_list)

    minibatches = []
    for start_index in start_list:
        minibatches.append(idx_list[start_index: start_index + len_sub_seq])

    minibatches_index = numpy.arange(len(minibatches)) # 切片的编号

    return zip(numpy.arange(num_sub_seq), minibatches) 

def test_seq_batch_index():

    data_len = 20
    batch_size = 4
    overhead = 3

    print get_seq_minibatches_idx(data_len, batch_size, overhead, True)


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

    P = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(100.0)  ) # 状态的协方差矩阵
    
    Qw = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(100.0) )  # 输入噪声协方差矩阵， 
    
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

    update_Qw = [(Qw,  0.9 * Qw)]

    update_W.extend(update_Qw)

    f_train = theano.function([x, y], T.dot(a, a.T), updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode,
                                    on_unused_input='warn')

    return f_train, P


def test_EKF():
    mu = 0
    sigma = 0.1

    n_input = 7
    n_output = 9

    dtype=theano.config.floatX


    W_in_data = numpy.random.normal(size=(n_input, n_output), loc=mu, scale=sigma)
    b_in_data = numpy.random.normal(size=(n_output,), loc=mu, scale=sigma)

    W_in = theano.shared(W_in_data.astype(dtype), name='W_in')              
    b_in = theano.shared(b_in_data.astype(dtype), name="b_in")

    W = []
    W.extend([W_in])
    W.extend([b_in])

    x_in = T.vector()
    y_out = T.tanh(T.dot(x_in,W_in) + b_in)
    y = T.vector()

    W_true = numpy.random.normal(size=(n_input, n_output))
    b_true = numpy.random.normal(size=(n_output))
    sampleNum = 500

    x_data = numpy.random.normal(size=(sampleNum,n_input))
    y_data = numpy.zeros(shape=(sampleNum,n_output))
    for i in range(sampleNum):
        y_data[i] = numpy.tanh(numpy.dot(x_data[i], W_true) + b_true)

    f_train, P = extend_kalman_train(W, y_out, n_output, y, x_in)

    for i in range(sampleNum):
        f_train(x_data[i],y_data[i])

    print W_true - W_in.get_value()
    print b_true - b_in.get_value()
    print 'P'
    print P.get_value()

if __name__ == '__main__':
    # test_seq_batch_index()
    test_EKF()





