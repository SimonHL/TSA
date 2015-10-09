# -*- coding: utf-8 -*-

import numpy
import theano

dtype=theano.config.floatX

theano.config.exception_verbosity = 'high'

def step(x,mask,h,WI):
	result = h + theano.dot(WI,x)
	result = result * mask
	return result

# 加要处理的数据
data = numpy.genfromtxt("mytestdata.txt")
data_x = data[:,0]
data_y = data[:,1]

n_segment = 2
n_hidden = 6
n_input = 4

WI = theano.shared(numpy.random.uniform(size=(n_hidden,n_input), 
	                                    low= -0.01, high=0.01).astype(dtype),
	               name='WI')

WH = theano.shared(numpy.random.uniform(size=(n_hidden,n_hidden),
	                                    low=-0.01, high=0.01).astype(dtype),
                   name='WH')

x = theano.tensor.matrix()
mask = theano.tensor.matrix()
h_init = theano.shared(numpy.zeros((1,n_hidden),dtype=dtype), name='h_init')

h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                             sequences=[x,mask],
                             outputs_info=h_init,
                             non_sequences=[WI])









