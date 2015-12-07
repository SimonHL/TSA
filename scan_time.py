# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

dtype=theano.config.floatX

theano.config.exception_verbosity = 'low'

def step(x,h):
	result = h + x
	return result

# 加要处理的数据
data = numpy.genfromtxt("mytestdata.txt")
data_x = data[:,0]
data_y = data[:,1]

n_segment = 2
n_hidden = 2
n_input = 4

WI = theano.shared(numpy.random.uniform(size=(n_hidden,n_input), 
	                                    low= -0.01, high=0.01).astype(dtype),
	               name='WI')

WH = theano.shared(numpy.random.uniform(size=(n_hidden,n_hidden),
	                                    low=-0.01, high=0.01).astype(dtype),
                   name='WH')

x = theano.tensor.vector()
H = theano.tensor.matrix()
h_init = theano.shared(numpy.zeros((1,n_hidden),dtype=dtype), name='h_init')

output_taps = [-1]
h_tmp, updates = theano.scan(step,  # 计算BPTT的函数
                             sequences=x,
                             outputs_info=dict(initial = H, taps=output_taps))

y = T.sum(h_tmp)

test_func = theano.function([x], [y,h_tmp], givens=[(H, h_init)]) 

data_x = numpy.arange(5)

for i in numpy.arange(2):
    tmp_y,tmp_h = test_func(data_x)
    print tmp_y, tmp_h
    print h_init.get_value()
    h_init.set_value(tmp_h[-1]+0.1)
    print 'continue'
     










