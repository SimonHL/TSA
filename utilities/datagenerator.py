# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class PublicFunction(object):
    
    @staticmethod
    def numpy_floatX(data):
        '''
        numpy数据类型和theano统一
        '''
        return numpy.asarray(data, dtype=theano.config.floatX)

    @staticmethod
    def data_get_data_x_y(seq_data, overhead):
        '''
        按照延时嵌入定理将seq_data加工成输入序列和输出序列
        overall为输入的维数
        '''
        data_x = seq_data[:-1]        # 最后一个不参与
        data_y = seq_data[overhead:]
        return data_x, data_y

    @staticmethod
    def adadelta(lr, tparams, grads, v, cost):
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
        v: Theano variable list. eg: [x,y]
        cost: Theano variable
            Objective fucntion to minimize

        Notes
        -----
        For more information, see [ADADELTA]_.

        .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
           Rate Method*, arXiv:1212.5701.
        """

        zipped_grads = [theano.shared(p.get_value() * PublicFunction.numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * PublicFunction.numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * PublicFunction.numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
                     
        updates_1 = zgup + rg2up

        f_grad_shared = theano.function(v, cost, updates=updates_1,
                                        name='adadelta_f_grad_shared',
                                        mode='FAST_RUN')

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
                                   mode='FAST_RUN')

        return updates_1, updates_2,f_grad_shared, f_update 

    @staticmethod
    def extend_kalman_train(W, y_hat, dim_y_hat, y):
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

        P = theano.shared( numpy.eye(dim_Wv) * PublicFunction.numpy_floatX(10.0) ) # 状态的协方差矩阵
        
        Qw = theano.shared( numpy.eye(dim_Wv) * PublicFunction.numpy_floatX(10.0) )  # 输入噪声协方差矩阵， 
        
        Qv = theano.shared( numpy.eye(dim_y_hat) * PublicFunction.numpy_floatX(0.01) )  # 观测噪声协方差矩阵

        # 求线性化的B矩阵: 系统输出y_hat对状态的一阶导数        
        params = []
        params.extend([y_hat])
        params.extend(W)
        def _step(*args):
            i = args[0]
            y_hat = args[1]
            params = args[2:]

            grads = T.grad(y_hat[i], params)

            tmp=[]
            for _g in grads:
                tmp.append(_g.flatten())
            return tmp

        B, updates = theano.scan(_step, 
                                sequences=T.arange(y_hat.shape[0]), 
                                non_sequences=params)

        B = T.concatenate(B, axis=1)

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

        cost = T.dot(a, a.T)

        return update_W, P, Qw, Qv, cost

class DataPrepare(object):
    @staticmethod
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

class Generator(object):
    """docstring for ClassName"""
    def __init__(self):
        self.data_generator = {
        0:self.load_data,
        1:self.data_dynamic_gen_1,
        2:self.data_dynamic_gen_2,
        3:self.data_gen_y_sin,
        'mackey_glass':self.data_mackey_glass
        }

    def load_data(self):
        data = numpy.genfromtxt("mytestdata.txt")
        data_x = data[:,0]
        data_y = data[:,1]
        return data_x, data_y
    
    def data_gen_y_sin(self):
        length = 1000
        data_x = numpy.zeros((length,))
        data_y = 1.0 * numpy.sin(20 * numpy.pi * numpy.linspace(0,1,length))
        return data_x, data_y

    def data_gen_x(self, n):
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

    def data_dynamic_gen_1(self):
        data_x = numpy.zeros((1000,))
        data_y = numpy.zeros((1000,),)
        for i in xrange(1000):
            data_x[i] = self.data_gen_x(i)
            if i > 2:
                tmp =  0.72 * data_y[i-1]
                tmp += 0.025*data_y[i-2] * data_x[i-1] 
                tmp += 0.001 * data_x[i-2]**2 
                tmp += 0.2 * data_x[i-3]
                data_y[i] = tmp

        return data_x, data_y

    def data_dynamic_gen_2(self):
        data_x = numpy.zeros((1000,))
        data_y = numpy.zeros((1000,),)
        for i in xrange(1000):
            data_x[i] = self.data_gen_x(i)
            if i > 2:
                tmp =  data_y[i-1] * data_y[i-2] * data_y[i-3] * data_x[i-1]
                tmp = tmp * (data_y[i-3] - 1) + data_x[i]
                tmp = tmp / (1 + data_y[i-2]**2 + data_y[i-3]**2)
                data_y[i] = tmp

        return data_x, data_y

    def data_mackey_glass(self):
        '''
        使用4阶龙格库塔计算mackey_glass吸引子
        dx/dt= -b * x(t) + a * x(t-tau) / ( 1 + x(t-tau)^10 )
        '''
        a = 0.2
        b = 0.1 
        tau = 30

        sample_time = 6000
        h = 0.1  # step
        t = numpy.arange(0, sample_time, h)
        
        N = len(t)
        x = numpy.zeros((N,))

        x[0] = 0.9   # initial value
        def _frac_part(_x):
            return a * _x / (1 + numpy.power(_x,10))
        for i in  numpy.arange(N-1):
            if t[i] < tau:
                k1 = -b * x[i]
                k2 = -b * (x[i] + h/2.0 * k1)
                k3 = -b * (x[i] + h/2.0 * k2)
                k4 = -b * (x[i] + h * k3)
                x[i+1]=x[i]+(k1+2*k2+2*k3+k4)*h/6
            else:
                n = i - numpy.floor(tau / h)
                k1 = _frac_part(x[n]) -b * x[i]
                k2 = _frac_part(x[n]) -b * (x[i] + h/2.0 * k1)
                k3 = _frac_part(x[n]) -b * (x[i] + h/2.0 * k2)
                k4 = _frac_part(x[n]) -b * (x[i] + h * k3)
                x[i+1]=x[i]+(k1+2*k2+2*k3+k4)*h/6

        resample_index = numpy.arange(0, N, int(6/h))  #以6s为周期重采样

        x = x[resample_index]
        return x, x

    def get_data(self,data_type):
        return self.data_generator.get(data_type)()


def main():
    '''
    模块测试代码
    '''
    data = Generator()
    x,y = data.get_data('mackey_glass')
    
    plt.subplot(211)
    plt.plot(numpy.arange(x.shape[0]), x, 'g')
    plt.subplot(212)
    plt.plot(numpy.arange(y.shape[0]), y, 'b')
    plt.show()

def test_data_prepare():
    data_len = 20
    batch_size = 4
    overhead = 3

    print DataPrepare.get_seq_minibatches_idx(data_len, batch_size, overhead, True)


if __name__ == '__main__':
    #main()
    test_data_prepare()