from keras.preprocessing import sequence
import numpy as np
np.random.seed(7)

def gen_data(x_data,y_data,TestLength,TrainLength):
    x_data=x_data[TestLength+TrainLength,:]
    y_data=y_data[TestLength+TrainLength,:]
    x_test_1 = x_data[0:TestLength, :]
    x_train_1=x_data[TestLength:TrainLength+TestLength, :]
    y_test_1=y_data[0:TestLength, :]
    y_train_1=y_data[TestLength:TrainLength+TestLength, :]
    np.savetxt('/data/biantian/data/'+str(x_test_1)+'_'+str(TestLength)+'.txt', x_test_1)
    np.savetxt('/data/biantian/data/' + str(x_train_1)+'_'+str(TrainLength) + '.txt', x_train_1)
    np.savetxt('/data/biantian/data/' + str(y_test_1)+'_'+str(TestLength) + '.txt', y_test_1)
    np.savetxt('/data/biantian/data/' + str(y_train_1)+'_' +str(TrainLength)+ '.txt', y_train_1)

    x_test_2 = x_data[TestLength:TestLength*2, :]
    x_train_2=np.delete(x_data,range(TestLength,TestLength*2))
    y_test_2=y_data[TestLength:TestLength*2, :]
    y_train_2=np.delete(y_data,range(TestLength,TestLength*2))
    np.savetxt('/data/biantian/data/' + str(x_test_2)+'_'+str(TestLength) + '.txt', x_test_2)
    np.savetxt('/data/biantian/data/' + str(x_train_2)+'_'+str(TrainLength) + '.txt', x_train_2)
    np.savetxt('/data/biantian/data/' + str(y_test_2)+'_'+str(TestLength) + '.txt', y_test_2)
    np.savetxt('/data/biantian/data/' + str(y_train_2)+'_'+str(TrainLength) + '.txt', y_train_2)

    x_test_3 = x_data[TestLength*2:TestLength*3, :]
    x_train_3=np.delete(x_data,range(TestLength*2,TestLength*3))
    y_test_3=y_data[TestLength*2:TestLength*3, :]
    y_train_3=np.delete(y_data,range(TestLength*2,TestLength*3))
    np.savetxt('/data/biantian/data/' + str(x_test_3)+'_'+str(TestLength) + '.txt', x_test_3)
    np.savetxt('/data/biantian/data/' + str(x_train_3)+'_'+str(TrainLength) + '.txt', x_train_3)
    np.savetxt('/data/biantian/data/' + str(y_test_3)+'_'+str(TestLength) + '.txt', y_test_3)
    np.savetxt('/data/biantian/data/' + str(y_train_3)+'_'+str(TrainLength) + '.txt', y_train_3)

    x_test_4 = x_data[TestLength*3:TestLength*4, :]
    x_train_4=x_data[0:TrainLength*3, :]
    y_test_4=y_data[TestLength*3:TestLength*4, :]
    y_train_4=y_data[0:TrainLength*3, :]
    np.savetxt('/data/biantian/data/' + str(x_test_4)+'_'+str(TestLength) + '.txt', x_test_4)
    np.savetxt('/data/biantian/data/' + str(x_train_4)+'_'+str(TrainLength) + '.txt', x_train_4)
    np.savetxt('/data/biantian/data/' + str(y_test_4)+'_'+str(TestLength) + '.txt', y_test_4)
    np.savetxt('/data/biantian/data/' + str(y_train_4)+'_'+str(TrainLength)+ '.txt', y_train_4)
    return

maxcount=95
#给数据打乱
y_data = np.loadtxt('/data/biantian/data/labels.txt')
filename = open('/data/biantian/data/filename.txt', encoding='utf-8')
filename=filename.read().split("\n")
filename=filename[0:4664]
index = [i for i in range(len(filename))]
np.random.shuffle(index)
y_data=y_data[index]
indices=0

filelist=[0]*len(filename)
for tempindex in index:
    filelist[indices] = filename[tempindex]
    indices=indices+1
Test_index = 0
for filename in filelist:
    print("导入训练集："+str(Test_index+1)+"/"+str(len(filelist)))
    new_x=np.loadtxt('/data/biantian/data/np_tfidf_dataset/' + str(filename))
    if (new_x.shape[0] == 5000):
        print(filename)
        # 只有一个时间间隔的文件
        # 3574766055179448.jsonwords.txt
        # 3493025545786832.jsonwords.txt
        new_x = new_x.reshape(1, 1, new_x.shape[0])
    else:
        new_x = new_x.reshape(1, new_x.shape[0], new_x.shape[1])
    new_x=sequence.pad_sequences(new_x, maxlen=maxcount, padding='post')
    if Test_index==0:
        x_data=new_x
    else:
        x_data = np.concatenate((x_data, new_x),axis=0)
    Test_index = Test_index + 1
gen_data(x_data,y_data,1000,3000)
gen_data(x_data,y_data,1120,3360)
gen_data(x_data,y_data,1050,3150)

