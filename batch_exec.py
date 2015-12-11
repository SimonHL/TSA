# -*- coding: utf-8 -*-

import os
import EKF_RNN
import IDRNN_EKF
import numpy

reload(EKF_RNN)
reload(IDRNN_EKF)

n_input=10
n_hidden=7
n_output=1
n_epochs=5
noise = 1
P0 = 10
Qw0 = 10

rnn =    EKF_RNN.RNN(      n_input=n_input, n_hidden=n_hidden, n_output=n_output)
rnn_c =  EKF_RNN.RNN(      n_input=n_input, n_hidden=n_hidden, n_output=n_output, continue_train=True)
id_rnn = IDRNN_EKF.IDRNN(  n_input=n_input, n_hidden=n_hidden, n_output=n_output)
id_rnn_c = IDRNN_EKF.IDRNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output, continue_train=True)

models = []
models.append(rnn)
# models.append(rnn_c)
models.append(id_rnn)
# models.append(id_rnn_c)

for m in models:
    m.build_model()

for i in xrange(20):
    numpy.random.seed()
    # SEED = int(numpy.random.randint(1000))
    SEED = i
    print 'Batch', i ,'begin!'

    for m in models:
        m.train(SEED,n_epochs, noise, P0, Qw0)

    print 'Batch', i ,'finished!'
print 'END!'