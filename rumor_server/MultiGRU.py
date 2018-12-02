from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import GRU
from keras.utils import np_utils
from keras.optimizers import Adagrad
import os
import numpy as np
     # same as the height of the image
np.random.seed(7)  # for reproducibility
LR = 0.5
maxcount=95
TrainLength=3000
TestLength=1000
batch=10

filedir='/data/biantian/data/dataset'
def generate_arrays_from_file(path,batch_size):
    global filelist
    global y_data
    while 1:
        #print(totalcnt)
        x_data = []
        index = 0
        count = [0] * batch_size
        totalcnt = 0
        X = []
        Y = []
        #for filename in os.listdir(path)[0:3000]:
        # if TestLength+totalcnt==TrainLength+TestLength:
        #     totalcnt=0
        for filename in filelist[TestLength:TestLength+TrainLength]:
            fx = open(str(path + '/' + filename), encoding='utf-8')
            x = fx.read().replace("[", "").replace("]", "").replace("\n", "\\\\")
            x = x.split("\\\\")
            x.pop()
            new_x = []
            for cur_x in x:
                count[index] = count[index] + 1
                cur_x = cur_x.split(",")
                temp_x = []
                for floatx in cur_x:
                    temp_x.append(float(floatx))
                new_x.append(temp_x)
            if len(new_x) == 0:
                continue
            x_data.append(new_x)
            index = index + 1
            #print("x:",index)

            if index == batch_size:
                # 通过补充-1将长度统一
                # X_data=[]
                tempx = 0
                global maxcount
                maxcount = max(max(count), maxcount)
                count = [0] * batch_size
                #print("training maxcount:", maxcount)
                for xdata in x_data:
                    tempx = tempx + 1
                    #print("train_x:", tempx)
                    if len(xdata) < maxcount:
                        for i in range(maxcount - len(xdata)):
                            xdata.append([-1] * 5000)
                    X_data_temp = np.array(xdata)
                    X_data_temp = X_data_temp.reshape(1, X_data_temp.shape[0], X_data_temp.shape[1])
                    if tempx == 1:
                        X_data = X_data_temp
                        continue
                    X_data = np.vstack((X_data, X_data_temp))
                    # X_data.append(xdata)
                # X_data=np.array(X_data)
                # x_train, x_test = X_data[0:1000, :], X_data[1000:1500, :]
                x_data=[]
                X = X_data
                Y= y_data[TestLength+totalcnt:TestLength+totalcnt+batch_size, :]
                totalcnt = totalcnt + index
                index = 0
                yield (X, Y)
                X = []
                Y = []
                #break
            fx.close()
    #     fy = open('/data/biantian/data/Weibo.txt', encoding='utf-8')
    #     y = fy.read().replace("\n", "\\\\")
    #     y = y.split("\\\\")
    #     y_data = []
    #     cnt=0
    #     #print(TestLength+totalcnt)
    #     #print(TrainLength+TestLength)
    #     #print(y[TestLength+totalcnt:TrainLength+TestLength])
    #     for cur_y in y[TestLength+totalcnt:TrainLength+TestLength]:
    #         #if tempy>3000:
    #         # if tempy <= TestLength+totalcnt:
    #         #     print("test1")
    #         #     continue
    #         # if tempy>TrainLength+TestLength:
    #         #     print("test2")
    #         #     break
    #         #print("train_y:", tempy)
    #         cur_y = cur_y.split(" ")
    #         new_y = cur_y[0].split("\t")
    #         if len(new_y) == 3:
    #             cnt = cnt + 1
    #             new1_y = new_y[1].split(":")
    #             y_data.append(new1_y[1])
    #         if cnt == batch_size:
    #             totalcnt=totalcnt+cnt
    #             Y = np_utils.to_categorical(y_data[0:batch_size], num_classes=2)
    #             yield (X, Y)
    #             X = []
    #             Y = []
    #             break
    # fy.close()

#读出所有y标签
f1 = open('/data/biantian/data/Weibo.txt', encoding='utf-8')
y = f1.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data = []
#tempy = 0
for cur_y in y:
    # tempy = tempy + 1
    # #if tempy<=TrainLength:
    # #    continue
    # #if tempy>TrainLength+TestLength:
    # if tempy > TestLength:
    #     break
    #print("test_y:", tempy)
    cur_y = cur_y.split(" ")
    new_y = cur_y[0].split("\t")
    if len(new_y) == 3:
        new1_y = new_y[1].split(":")
        y_data.append(new1_y[1])
        #print(new1_y[1])
f1.close()
temp_y_data=y_data
#给数据打乱
index = [i for i in range(len(os.listdir(filedir)))]
np.random.shuffle(index)
indices=0
filelist=[0]*len(os.listdir(filedir))
new_y_data=[0]*len(os.listdir(filedir))
for tempindex in index:
    filelist[indices] = os.listdir(filedir)[tempindex]
    new_y_data[indices] = temp_y_data[tempindex]
    indices=indices+1
#定义测试集标签
y_data = np_utils.to_categorical(new_y_data, num_classes=2)
y_test = y_data[0:TestLength, :]

Test_x_data = []
Test_index = 0
Test_count = [0] * TestLength
for filename in filelist[0:TestLength+1]:
    f = open(str(filedir + '/' + filename), encoding='utf-8')
    x = f.read().replace("[", "").replace("]", "").replace("\n", "\\\\")
    x = x.split("\\\\")
    x.pop()
    new_x = []
    for cur_x in x:
        Test_count[Test_index] = Test_count[Test_index] + 1
        cur_x = cur_x.split(",")
        temp_x = []
        for floatx in cur_x:
            temp_x.append(float(floatx))
        new_x.append(temp_x)
    f.close()
    if len(new_x) == 0:
        continue
    Test_x_data.append(new_x)
    Test_index = Test_index + 1
    #print(Test_index)
    if Test_index == TestLength:
        break
# 通过补充-1将长度统一
# X_data=[]
tempx = 0
#print("Test_x:",len(Test_x_data))
#print(Test_x_data[1][1])
for xdata in Test_x_data:
    tempx = tempx + 1
    if len(xdata) < maxcount:
        for i in range(maxcount - len(xdata)):
            xdata.append([-1] * 5000)
    X_data_temp = np.array(xdata)
    X_data_temp = X_data_temp.reshape(1, X_data_temp.shape[0], X_data_temp.shape[1])
    if tempx == 1:
        X_data = X_data_temp
        continue
    X_data = np.vstack((X_data, X_data_temp))
    # X_data.append(xdata)
# X_data=np.array(X_data)
x_test = X_data[0:TestLength, :]
maxcount=max(max(Test_count),maxcount)
#print("test maxcount:",maxcount)


model = Sequential()
model.add(Masking(mask_value= -1,input_shape=(maxcount, 5000,)))
#model.add(Embedding(5000,100))
#model.add(Flatten())
model.add(Dense(100))
model.add(GRU(100,return_sequences=True))
model.add(GRU(100))
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file(filedir,batch_size=batch),validation_data=(x_test, y_test),epochs=10,steps_per_epoch=300)
#model.fit(x_train, y_train, batch_size=100, epochs=10)
#score,acc = model.evaluate(x_test, y_test, batch_size=batch)
#print('score:', score)
#print('acc:', acc)
