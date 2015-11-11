# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T


compile_mode = 'FAST_COMPILE'
# compile_mode = 'FAST_RUN'

# Set the random number generators' seeds for consistency
SEED = 200
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

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


def extend_kalman_train(W, y_hat, y, x):
    '''
    P: 状态对应的协方差矩阵
    Qv: 观测噪声的协方差矩阵
    Qw: 输入噪声的协方差矩阵
    W: 需要估计的状态
    y_hat: 系统的输出
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

    dim_y_hat = n_output

    print dim_Wv, dim_y_hat
    print W_vec

    P = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0)  ) # 状态的协方差矩阵
    
    Qw = theano.shared( numpy.eye(dim_Wv) * numpy_floatX(10.0) )  # 输入噪声协方差矩阵， 
    
    Qv = theano.shared( numpy.eye(dim_y_hat) * numpy_floatX(0.01) )  # 观测噪声协方差矩阵


    # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数
    B = []
    for _W in W:
        J, updates = theano.scan(lambda i, y_hat, W: T.grad(y_hat[i][0], _W).flatten(), 
                                 sequences=T.arange(y_hat.shape[0]), 
                                 non_sequences=[y_hat, _W])
        B.extend([J])

    B = T.concatenate(tuple(B),axis=1)

    # 计算残差
    a = y - y_hat

    # 计算增益矩阵
    G = T.dot(T.dot(P,B.T), T.nlinalg.matrix_inverse(T.dot(T.dot(B,P),B.T)+Qv)) 

    # 计算新的状态
    update_W_vec = W_vec  +  (T.dot(G, a)).T



    # 计算新的状态协方差阵
    delta_P = -T.dot(T.dot(G,B), P) + Qw 
    update_P = [(P, P + delta_P)] 

    # 逆矢量化
    bi = 0
    delta_W = []
    for i in numpy.arange(len(W)):
        be = bi+W[i].size
        delta_tmp = update_W_vec[0,bi:be]
        delta_W.append( delta_tmp.reshape(W[i].shape) )
        bi = be

    update_W = [ (_W, _dW) for (_W, _dW) in  zip(W, delta_W) ]

    update_W.extend(update_P)

    update_Qw = [(Qw,  1.0* Qw)]

    update_W.extend(update_Qw)

    f_train = theano.function([x, y], T.dot(a, a.T), updates=update_W,
                                    name='EKF_f_train',
                                    mode=compile_mode,
                                    on_unused_input='warn')

    return f_train


mu = 0
sigma = 0.1

n_input = 4
n_output = 1

dtype=theano.config.floatX

W = []

W_in_data = numpy.random.normal(size=(n_output, n_input), loc=mu, scale=sigma)
b_in_data = numpy.random.normal(size=(n_output,1), loc=mu, scale=sigma)

W_in = theano.shared(W_in_data.astype(dtype), 
                      name='W_in')              
b_in = theano.shared(b_in_data.astype(dtype), name="b_in")

W.extend([W_in])
W.extend([b_in])

x_in = T.col()
y_out = T.tanh(T.dot(W_in, x_in) + b_in)
y = T.col()

W_true = numpy.array([[0.1, 0.5,  0.2,  -0.1]])
b_true = numpy.array([0.03])
sampleNum = 50


x_data = numpy.random.normal(size=(sampleNum,4,1))
y_data = numpy.zeros(shape=(sampleNum,n_output, 1))
for i in range(sampleNum):
    y_data[i] = numpy.tanh(numpy.dot(W_true,x_data[i]) + b_true)

f_train = extend_kalman_train(W, y_out, y, x_in)

for i in range(sampleNum):
    print f_train(x_data[i],y_data[i])

print W_in.get_value()
print b_in.get_value()



