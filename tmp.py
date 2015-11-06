# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

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

n = 10
minibatch_size = 10
overhead = 2

minib = get_minibatches_idx(n, minibatch_size, overhead)
print minib