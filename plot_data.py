# -*- coding: utf-8 -*-

import numpy
import os
import matplotlib.pyplot as plt

files = os.listdir(os.getcwd())
# files = ['MaskRNN_b1_i0_h25_nh0_S80.npz'] 

b_plot_single = False

matched_num = numpy.zeros((6))
matched_err_sum = {}
history_err_sum = {}

for file_name in files:
    i = file_name.find('MaskRNN_b')
    if i == -1:
        continue
    if b_plot_single:
        seed_number = file_name.find('9._') 
        if seed_number == -1:
            continue
    npzfile = numpy.load(file_name)
    bname = file_name[9]
    b = int(bname)
    if matched_num[b] == 0:
        matched_err_sum[b] = npzfile['cumulative_error']
        history_err_sum[b] = npzfile['history_errs']
    else:
        matched_err_sum[b] += npzfile['cumulative_error']
        history_err_sum[b] += npzfile['history_errs']
    matched_num[b] += 1

print 'matched_num: '
print matched_num

colors = ['r', 'g', 'b', 'y','k','m']
print 'colors:', colors

plt.figure(1)
for (k,v) in matched_err_sum.items():
    data = v / matched_num[k]
    plt.plot(numpy.arange(data.shape[0]), data, colors[k])
    plt.grid(True)

plt.figure(2)     
for (k,v) in history_err_sum.items():
    data = v[1:,1] / matched_num[k]  # average valid_err
    plt.plot(numpy.arange(data.shape[0]), data, colors[k])
    plt.grid(True)

plt.show()






