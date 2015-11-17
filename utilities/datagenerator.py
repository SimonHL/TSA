# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt

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