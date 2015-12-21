# -*- coding: utf-8 -*-
"""
使用Elman网络（简单局部回归网络）

@author: simon
"""
import sys,time
import getopt
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy

import utilities.datagenerator as DG
reload(DG)

compile_mode = 'FAST_COMPILE'
theano.config.exception_verbosity = 'low'
dtype=theano.config.floatX

class RNN(object):
    def __init__(self, 
                 build_method=0, # 0: RNN
                 init_method=0,  # 0: normal   1: uniform
                 n_input=7,n_hidden=5,n_output=1,
                 batch_size=1,
                 continue_train=False):

        # 设置网络参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        

        self.n_predict = 150

        self.continue_train = continue_train

        if continue_train:
            build_method = 1
        self.build_method = build_method
        self.init_method = init_method
        self.batch_size = batch_size 
        self.patience = 100
        self.valid_fre = 20
        
        self.h_init = theano.shared(numpy.zeros((1,n_hidden), dtype=dtype), name='h_init') # 网络隐层初始值

        mu,sigma = 0.0, 0.1
        if init_method == 0:
            self.W_in = [theano.shared(numpy.random.normal(size=(1, n_hidden),
                                  loc=mu, scale=sigma).astype(dtype), 
                                  name='W_in' + str(u)) for u in range(n_input)]                
            self.b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
            self.W_hid = theano.shared(numpy.random.normal(size=(n_hidden, n_hidden), 
                                  loc=mu, scale=sigma).astype(dtype), name='W_hid') 
            self.W_out = theano.shared(numpy.random.normal(size=(n_hidden,n_output),
                                  loc=mu,scale=sigma).astype(dtype),name="W_out")
            self.b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out")
        else:
            self.W_in = [theano.shared(numpy.random.uniform(size=(1, n_hidden), 
                                  low=-0.01, high=0.01).astype(dtype), 
                                  name='W_in' + str(u)) for u in range(n_input)]                
            self.b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name="b_in")
            self.W_hid = theano.shared(numpy.random.uniform(size=(n_hidden, n_hidden), 
                                  low=-0.01, high=0.01).astype(dtype), name='W_hid') 
            self.W_out = theano.shared(numpy.random.uniform(size=(n_hidden,n_output),
                                  low=-0.01,high=0.01).astype(dtype),name="W_out")
            self.b_out = theano.shared(numpy.zeros((n_output,), dtype=dtype),name="b_out") 
    
    def set_init_parameters(self, SEED, P0, Qw0):
        numpy.random.seed(SEED)
        mu,sigma = 0.0, 0.1
        for i in self.W_in:
            i.set_value(numpy.random.normal(size=(1, self.n_hidden), loc=mu, scale=sigma))

        self.b_in.set_value( numpy.zeros((self.n_hidden,), dtype=dtype))
        self.W_hid.set_value(numpy.random.normal(size=(self.n_hidden, self.n_hidden), loc=mu, scale=sigma))
        # self.W_hid.set_value(numpy.eye(self.n_hidden))
        self.W_out.set_value(numpy.random.normal(size=(self.n_hidden, self.n_output),  loc=mu, scale=sigma))
        self.b_out.set_value(numpy.zeros((self.n_output,), dtype=dtype))

        self.h_init.set_value(numpy.zeros((1,self.n_hidden), dtype=dtype))

        self.P.set_value(numpy.eye(self.P.get_value().shape[0]) * numpy.asarray(P0, dtype=dtype))
        self.Qw.set_value(numpy.eye(self.Qw.get_value().shape[0])* numpy.asarray(Qw0, dtype=dtype))
        self.Qv.set_value(numpy.eye(self.Qv.get_value().shape[0])* numpy.asarray(0.01, dtype=dtype))
    def step(self, *args):
        x =  [args[u] for u in xrange(self.n_input)] 
        hid_taps = args[self.n_input]    
        
        h = T.dot(x[0], self.W_in[0])
        for j in xrange(1, self.n_input):           # 前向部分
            h +=  T.dot(x[j], self.W_in[j])
        
        h += T.dot(hid_taps, self.W_hid)            # 回归部分
        h += self.b_in                              # 偏置部分
        h = T.tanh(h)

        y = T.dot(h,self.W_out) + self.b_out        # 线性输出
        return h, y
   
    def  gen_drive_sin(self,sampleNum,N):
        '''
        生成一个长度为sampleNum, 周期为N的正弦信号
        '''
        data = 1.0 * numpy.sin(2 * numpy.pi / N  * numpy.arange(sampleNum))
        return data

    def prepare_data(self, data_x, data_mask, data_y):
        '''
        将数据分为训练集，验证集和测试集

        注意，因为要进行hstack, 行向量会变为列向量
        '''
        data_len = len(data_y)
        train_end = numpy.floor(data_len * 0.5)
        test_end = numpy.floor(data_len * 0.8)

        if data_x.ndim == 1:
            data_x.resize((data_x.shape[0],1))
        if data_mask != []  and data_mask.ndim == 1:
            data_mask.resize((data_mask.shape[0],1))
        if data_y.ndim == 1:
            data_y.resize((data_y.shape[0],1))

        if data_mask == []:
            allData = numpy.concatenate((data_x,data_y), axis=1)
        else:
            allData = numpy.concatenate((data_x,data_mask,data_y), axis=1)

        train_data = allData[:train_end,...]
        test_data = allData[train_end:test_end,...]
        valid_data = allData[test_end:,...]

        return train_data, valid_data, test_data

    def build_model(self):
        # 构造网络
        x_in = T.vector()   # 输入向量,第1维是时间
        y_out = T.vector()  # 输出向量
        lr = T.scalar()     # 学习速率，标量

        H = T.matrix()      # 隐单元的初始化值

        start_time = time.clock()  
        input_taps = range(1-self.n_input, 1)
        output_taps = [-1]
        [h_tmp,y], _ = theano.scan(self.step,  # 计算BPTT的函数
                                sequences=dict(input=x_in, taps=input_taps),  # 从输出值中延时-1抽取
                                outputs_info=[dict(initial = H, taps=output_taps), None])
                               
        y = T.flatten(y)
        
        params = []
        params.extend(self.W_in)
        params.extend([self.b_in])
        params.extend([self.W_hid])
        params.extend([self.W_out])   
        params.extend([self.b_out]) 

        if self.continue_train:
            ada_method = 1
        else:
            ada_method = 0
        update_W, self.P, self.Qw, self.Qv, cost = DG.PublicFunction.extend_kalman_train(params, y, self.batch_size, y_out, ada_method)

        self.f_train = theano.function([x_in, y_out], [cost, h_tmp[-self.batch_size]], updates=update_W,
                                        name='EKF_f_train',
                                        mode=compile_mode,
                                        givens=[(H, self.h_init)])                                              

        self.sim_fn = theano.function([x_in], outputs=y, givens=[(H, self.h_init)])
        self.pred_cost = theano.function([x_in, y_out], outputs=cost, givens=[(H, self.h_init)]) 

        print 'build time (%.5fs)' % ((time.clock() - start_time) / 1.)

    def train(self, SEED, n_epochs, noise, P0, Qw0):
        # 加要处理的数据
        g = DG.Generator()
        data_x,data_y = g.get_data('mackey_glass')
        # data_x,data_y = g.get_data('sea_clutter_lo')
        print data_x.shape
        noise_begin = int(data_x.shape[0] * 0.65)
        noise_end = int(data_x.shape[0] * 0.7)
        data_x[noise_begin:noise_end] += 0.1*self.gen_drive_sin(noise_end-noise_begin,10)
        normal_noise = numpy.random.normal(size=data_x.shape, loc=0, scale=0.02)
        # data_x += normal_noise
        plt.figure(123)
        plt.plot(normal_noise,'r')
        plt.plot(data_x,'b')

        data_y = data_x
        train_data, valid_data, test_data = self.prepare_data(data_x, [], data_y) # data_x 会成为列向量

        print 'train info:', train_data.shape
        print 'valid info:', valid_data.shape
        print 'test info:', test_data.shape
        self.history_errs = numpy.zeros((n_epochs*train_data.shape[0],3), dtype=dtype)  
        history_errs_cur_index= 0
        bad_counter = 0
        start_time = time.clock()   
        mu_noise, sigma_noise = 0, noise
        self.saveto = 'MaskRNN_b{}_i{}_h{}_nh{}_S{}._p{}.npz'.format(
                   self.build_method, self.init_method, self.n_hidden, sigma_noise, SEED,n_epochs)

        print 'Result will be saved to: ',self.saveto
        print "noise level:", mu_noise, sigma_noise 

        # 初始化参数
        self.set_init_parameters(SEED, P0, Qw0)

        for epochs_index in xrange(n_epochs) :  
            kf = DG.DataPrepare.get_seq_minibatches_idx(train_data.shape[0], self.batch_size, self.n_input, shuffle=False)
            for batch_index, train_index in kf:
                sub_seq = train_data[train_index,1] 
                _x, _y = DG.PublicFunction.data_get_data_x_y(sub_seq, self.n_input)
                train_err, h_init_continue = self.f_train(_x, _y)                
                if self.continue_train:
                    # sigma_noise = numpy.sqrt(numpy.max(self.Qw.get_value()))
                    noise_add = numpy.random.normal(size=(1,self.n_hidden), loc=mu_noise, scale=sigma_noise)
                    self.h_init.set_value(h_init_continue + noise_add)
                    # self.h_init.set_value(numpy.random.normal(size=(1,self.n_hidden), loc=0, scale=0.5))
                # else:
                #     self.h_init.set_value(h_init_continue)
                # print '{}.{}: online train error={:.6f}'.format(epochs_index, batch_index, float(train_err))

                if numpy.mod(batch_index+1, self.valid_fre) == 0:
                    train_err =  self.pred_cost(train_data[:-1,0], train_data[self.n_input:,1]) / train_data.shape[0]
                    test_err = self.pred_cost(test_data[:-1,0], test_data[self.n_input:,1])  / test_data.shape[0] 
                    valid_err = self.pred_cost(valid_data[:-1,0], valid_data[self.n_input:,1]) / valid_data.shape[0]
                    
                    print '{}: train error={:.6f}, valid error={:.6f}, test error={:.6f}'.format(
                        epochs_index, float(train_err), float(valid_err), float(test_err))

                    self.history_errs[history_errs_cur_index,:] = [train_err, valid_err, test_err]
                    history_errs_cur_index += 1

                    if valid_err <= self.history_errs[:history_errs_cur_index,1].min():
                        bad_counter = 0

                    if history_errs_cur_index > self.patience and valid_err >= self.history_errs[:history_errs_cur_index-self.patience,1].min():
                        bad_counter += 1
                        if bad_counter > self.patience * train_data.shape[0]:
                            print 'Early Stop!'
                            break

        self.history_errs = self.history_errs[:history_errs_cur_index,:]

        # 计算多步误差
        x_train_end = train_data[-self.n_input:,0]
        if self.continue_train:
            self.h_init.set_value(h_init_continue)
        y_predict = numpy.zeros((self.n_predict,))
        cumulative_error = 0
        cumulative_error_list = numpy.zeros((self.n_predict,))
        for i in numpy.arange(self.n_predict):
            y_predict[i] = self.sim_fn(x_train_end)
            x_train_end[:-1] = x_train_end[1:]
            x_train_end[-1] = y_predict[i]
            cumulative_error += numpy.abs(y_predict[i] - test_data[i,1])
            cumulative_error_list[i] = cumulative_error

        # 计算整体的单步误差
        y_sim = self.sim_fn(data_x[:-1,0]) 
        print 'y_sim.shape: ', y_sim.shape


        # 保存结果
        numpy.savez(self.saveto, cumulative_error=cumulative_error_list,
            history_errs = self.history_errs)

        print 'Result have been saved to: ',self.saveto

        # plot 数据
        self.data_x = data_x
        self.data_y = data_y
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

        self.y_sim = y_sim
        self.y_predict = y_predict
        self.cumulative_error_list = cumulative_error_list

        print 'train time (%.5fs)' % ((time.clock() - start_time) / 1.)

    def plot_data(self):
        plt.figure(1)
        plt.plot(numpy.arange(self.n_predict), self.cumulative_error_list)
        plt.title('cumulative error')
        plt.grid(True)

        plt.figure(2)
        plt.plot(numpy.arange(self.y_predict.shape[0]), self.y_predict,'r')
        plt.plot(numpy.arange(self.y_predict.shape[0]), self.test_data[:self.y_predict.shape[0],-1],'g')

        plt.figure(3)
        index_start = self.data_x.shape[0]-self.y_sim.shape[0]
        index_train_end = self.train_data.shape[0]
        index_test_end = index_train_end + self.test_data.shape[0]
        index_valid_end = index_test_end + self.valid_data.shape[0]
        train_index = numpy.arange(index_train_end-index_start)
        test_index  = numpy.arange(index_train_end-index_start,index_test_end-index_start)
        valid_index = numpy.arange(index_test_end-index_start,index_valid_end-index_start)

        plt.plot(train_index, self.y_sim[train_index],'r')
        plt.plot(test_index, self.y_sim[test_index],'y')
        plt.plot(valid_index, self.y_sim[valid_index],'b')
        plt.plot(self.data_y[self.n_input:],'k')  # 原始信号
        plt.plot(self.y_sim-self.data_y[self.n_input:,0], 'g')

        plt.figure(4)
        plt.plot( self.history_errs[:,0], 'r')
        plt.plot( self.history_errs[:,1], 'g')
        plt.plot( self.history_errs[:,2], 'b')

        plt.show()

if __name__ == '__main__':
    try:  
        opts, args = getopt.getopt(sys.argv[1:], "pcs:i:h:o:n:", 
                                   ["plot","continue","seed=", "input=","hidden=","output=","epochs="])  
    except getopt.GetoptError:  
        print 'parameter Error! '
        sys.exit() 
    
    SEED = 29
    n_input=10
    n_hidden=7
    n_output=1
    n_epochs=5
    noise = 1
    P0 = 10
    Qw0 = 10

    b_plot = False
    continue_train = False

    for o, a in opts:
        if o in ("-p","--plot"):
            b_plot = True 
        if o in ("-c","--continue"):
            continue_train = True
        if o in ("-s", "--seed"):  
            SEED = int(a)  
        if o in ("-i", "--input"):  
            n_input = int(a)
        if o in ("-h", "--hidden"):  
            n_hidden = int(a)
        if o in ("-o", "--output"):  
            n_output = int(a) 
        if o in ("-n", "--epochs"):  
            n_epochs = int(a) 

    rnn = RNN( n_input=n_input, n_hidden=n_hidden, n_output=n_output, continue_train = continue_train)
    rnn.build_model()
    rnn.train(SEED, n_epochs,noise,P0,Qw0)
    if b_plot:
        rnn.plot_data()