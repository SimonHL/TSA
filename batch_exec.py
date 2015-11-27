# -*- coding: utf-8 -*-

import os

for i in xrange(20):
    # os.system('python MaskRNN_EKF.py')
    # os.system('python EKF_RNN.py')
    # os.system('python EKF_RNN_tmp_copy.py')
    # os.system('python EKF_LSTM.py')
    os.system('python EKF_RNN.py')
    os.system('python EKF_RNN_noise.py')
    os.system('python IDEKF_RNN.py')
    os.system('python IDEKF_RNN_noise.py')
    print 'Batch' + i + 'finished!', 
print 'END!'