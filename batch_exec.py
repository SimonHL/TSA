# -*- coding: utf-8 -*-

import os
import EKF_RNN
import IDRNN_EKF
import numpy

reload(EKF_RNN)
reload(IDRNN_EKF)

n_input=7
n_hidden=15
n_output=1
n_epochs=40

rnn =    EKF_RNN.RNN(      n_input=n_input, n_hidden=n_hidden, n_output=n_output, n_epochs=n_epochs)
rnn_c =  EKF_RNN.RNN(      n_input=n_input, n_hidden=n_hidden, n_output=n_output, n_epochs=n_epochs, continue_train=True)
id_rnn = IDRNN_EKF.IDRNN(  n_input=n_input, n_hidden=n_hidden, n_output=n_output, n_epochs=n_epochs)
id_rnn_c = IDRNN_EKF.IDRNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output, n_epochs=n_epochs, continue_train=True)

models = []
models.append(rnn)
models.append(rnn_c)
models.append(id_rnn)
models.append(id_rnn_c)

for m in models:
    m.build_model()

for i in xrange(50):
    SEED = int(numpy.random.lognormal()*100)
    print 'Batch', i ,'begin!'

    for m in models:
        m.train(SEED)

    print 'Batch', i ,'finished!'
print 'END!'