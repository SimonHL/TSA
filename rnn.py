import numpy
import theano
import theano.tensor as TT
import utilities.datagenerator as DG
import matplotlib.pyplot as plt

# number of hidden units
n = 5
# number of input units
nin = 7
# number of output units
nout = 1

# input (where first dimension is time)
u = TT.matrix()
# target (where first dimension is time)
t = TT.matrix()
# initial hidden state of the RNN
h0 = TT.vector()
# learning rate
lr = TT.scalar()
# recurrent weights as a shared variable
W = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_in = theano.shared(numpy.random.uniform(size=(nin, n), low=-.01, high=.01))
# hidden to output layer weights
W_out = theano.shared(numpy.random.uniform(size=(n, nout), low=-.01, high=.01))


# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h_tm1, W, W_in, W_out):
    h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
    y_t = TT.dot(h_t, W_out)
    return h_t, y_t

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])
# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPTT
gW, gW_in, gW_out = TT.grad(error, [W, W_in, W_out])
# training function, that computes the error and updates the weights using
# SGD.
fn = theano.function([h0, u, t, lr],
                     error,
                     updates=[(W, W - lr * gW),
                             (W_in, W_in - lr * gW_in),
                             (W_out, W_out - lr * gW_out)])

fn_sim = theano.function([h0,u], y)

g = DG.Generator()
data_x_,data_y_ = g.get_data('mackey_glass')
N = 1000
data_x_ = data_x_[:N]
data_y_ = data_y_[:N]
learning_rate = 0.0005
sampleNum = data_y_.shape[0]-nin  # how many group_x can be constructed
data_x = numpy.zeros((sampleNum,nin))
data_y = numpy.zeros((sampleNum,nout))

for i in numpy.arange(sampleNum):
    data_x[i] = data_y_[i:i+nin]
    data_y[i] = data_y_[i+nin]    

dtype = theano.config.floatX
h_init = numpy.zeros(shape=(n,), dtype=dtype)
n_epoch = 500
for i in numpy.arange(n_epoch):
    print '{}.{}: cost=: {} '.format(i,0,fn(h_init,data_x,data_y,learning_rate))

y_sim = fn_sim(h_init,data_x)
plt.plot(numpy.arange(y_sim.shape[0]), y_sim, 'r')
plt.plot(numpy.arange(y_sim.shape[0]), data_y, 'k')
plt.plot(numpy.arange(y_sim.shape[0]), y_sim - data_y, 'g')
plt.show()





