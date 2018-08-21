from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import Adagrad
import os
import numpy as np

     # same as the height of the image
np.random.seed(7)  # for reproducibility

INPUT_SIZE = 5000     # same as the width of the image
BATCH_SIZE = 1
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 100
#TIME_STEPS = 50
LR = 0.5

filedir='F:\论文汇总\谣言检测\数据\A16rumdect\dataset'
x_data=[]
index=0
count=[0]*200
for filename in os.listdir(filedir):
    f = open(str(filedir + '/' + filename),encoding='utf-8')
    x = f.read().replace("[","").replace("]","").replace("\n","\\\\")
    x=x.split("\\\\")
    x.pop()
    new_x=[]
    for cur_x in x:
        count[index]=count[index]+1
        cur_x=cur_x.split(",")
        temp_x=[]
        for floatx in cur_x:
            temp_x.append(float(floatx))
        new_x.append(temp_x)
    f.close()
    if len(new_x) == 0:
        continue
    x_data.append(new_x)
    index=index+1
    if index==200:
        break

f1 = open('F:\论文汇总\谣言检测\数据\A16rumdect\Weibo.txt', encoding='utf-8')
y = f1.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data=[]
for cur_y in y:
    cur_y=cur_y.split(" ")
    new_y=cur_y[0].split("\t")
    if len(new_y)==3:
        new1_y=new_y[1].split(":")
        y_data.append(new1_y[1])
f1.close()
y_data = np_utils.to_categorical(y_data[0:200], num_classes=2)
y_train,y_test=y_data[0:150,:],y_data[150:200,:]


model = Sequential()
model.add(Embedding(5000,100))
model.add(LSTM(100))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
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

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

"""
model.fit(X_train, y_train, batch_size=1, epochs=100)
score = model.evaluate(X_test, y_test, batch_size=1)
print('score:', score)
"""

total_cost = 0
total_accuray = 0
for step in range(150):
    # data shape = (batch_num, steps, inputs/outputs)
    #X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    X_batch=x_data[step]
    X_batch=np.array(X_batch)
    #count=np.array(count)
    X_batch=X_batch.reshape(1,X_batch.shape[0],X_batch.shape[1])
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    #TIME_STEPS=count[step]
    #cost = model.train_on_batch([X_batch,TIME_STEPS], Y_batch)
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    #BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    BATCH_INDEX = 0 if BATCH_INDEX >= 150 else BATCH_INDEX
    if (step+1)% 15==0:
        #print("xstep",step)
        print('test cost: ', total_cost/5, 'test accuracy: ', total_accuray/5)
        total_cost=0
        total_accuray=0
    if step % 3 == 0 and step>0:#相当于5个一个batch
        y_STEPS=int(step/3+149)
        X_test = np.array(x_data[y_STEPS])
        X_test = X_test.reshape(1,X_test.shape[0], X_test.shape[1])
        Y_test=y_test[y_STEPS-150]
        #print("ystep",y_STEPS-150)
        Y_test=Y_test.reshape(1, Y_test.shape[0])
        #cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=1, verbose=False)
        total_cost=total_cost+cost
        total_accuray=total_accuray+accuracy
        #print('test cost: ', cost, 'test accuracy: ', accuracy)