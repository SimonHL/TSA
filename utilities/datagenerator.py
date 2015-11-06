# -*- coding: utf-8 -*-

import numpy

class Generator(object):
    """docstring for ClassName"""
    def __init__(self):
        self.data_generator = {
        0:self.load_data,
        1:self.data_dynamic_gen_1,
        2:self.data_dynamic_gen_2,
        3:self.data_gen_y_sin
        }

    def load_data(self):
        data = numpy.genfromtxt("..\mytestdata.txt")
        data_x = data[:,0]
        data_y = data[:,1]
        return data_x, data_y
    
    def data_gen_y_sin(self):
        length = 1000
        data_x = numpy.zeros((length,))
        data_y = 1.0 * numpy.sin(30 * numpy.pi * numpy.linspace(0,1,length))
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

    def get_data(self,data_type):
        return self.data_generator.get(data_type)()


def main():
    '''
    模块测试代码
    '''
    data = Generator()
    x,y = data.get_data(3)
    print x.shape, y.shape


if __name__ == '__main__':
    main()