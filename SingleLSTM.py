from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
TIME_STEPS = 28     # same as the height of the image
INPUT_SIZE = 28     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Embedding(max_features, output_dim=256))

'''
LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs)
output_dim: 输出层的维数,或者可以用output_shape
init:
    uniform(scale=0.05) :均匀分布，最常用的。Scale就是均匀分布的每个数据在-scale~scale之间。此处就是-0.05~0.05。scale默认值是0.05；
    lecun_uniform:是在LeCun在98年发表的论文中基于uniform的一种方法。区别就是lecun_uniform的scale=sqrt(3/f_in)。f_in就是待初始化权值矩阵的行。
    normal：正态分布（高斯分布)。
    Identity ：用于2维方阵，返回一个单位阵.
    Orthogonal：用于2维方阵，返回一个正交矩阵. lstm默认
    Zero：产生一个全0矩阵。
    glorot_normal：基于normal分布，normal的默认 sigma^2=scale=0.05，而此处sigma^2=scale=sqrt(2 / (f_in+ f_out))，其中，f_in和f_out是待初始化矩阵的行和列。
    glorot_uniform：基于uniform分布，uniform的默认scale=0.05，而此处scale=sqrt( 6 / (f_in +f_out)) ，其中，f_in和f_out是待初始化矩阵的行和列。
W_regularizer , b_regularizer  and activity_regularizer:
    官方文档: http://keras.io/regularizers/
    from keras.regularizers import l2, activity_l2
    model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    加入规则项主要是为了在小样本数据下过拟合现象的发生,我们都知道,一半在训练过程中解决过拟合现象的方法主要中两种,一种是加入规则项(权值衰减), 第二种是加大数据量
    很显然,加大数据量一般是不容易的,而加入规则项则比较容易,所以在发生过拟合的情况下,我们一般都采用加入规则项来解决这个问题.
'''

model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(X_test, y_test, batch_size=16)