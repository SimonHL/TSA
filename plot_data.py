# -*- coding: utf-8 -*-

import numpy
import os
import matplotlib.pyplot as plt

files = os.listdir(os.getcwd())

# files = ['MaskRNN_b1_i0_h25_nh0_S80.npz'] 

matched_num = numpy.zeros((6))
matched_err_sum = {}

for file_name in files:
    i = file_name.find('MaskRNN_b')
    if i == -1:
        continue 
    npzfile = numpy.load(file_name)
    bname = file_name[9]
    b = int(bname)
    if matched_num[b] == 0:
        matched_err_sum[b] = npzfile['cumulative_error']
    else:
        matched_err_sum[b] += npzfile['cumulative_error']
    matched_num[b] += 1

print 'matched_num: '
print matched_num

colors = ['r', 'g', 'b', 'y','k','m']
print 'colors:', colors
for (k,v) in matched_err_sum.items():
     print "build_method:", k
     data = v / matched_num[k]
     plt.plot(numpy.arange(data.shape[0]), data, colors[k])

plt.grid(True)
plt.show()






