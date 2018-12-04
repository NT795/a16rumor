# -*- coding: utf-8 -*-
import json
import os
import math
import heapq
from sklearn import feature_extraction
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def resolveJson(path):
    file = open(path, "r")
    fileJson = json.load(file)
    ori_length=len(fileJson)
    length=ori_length
    for index in range(ori_length):
        if fileJson[index]["text"]=='':
            length=length-1
            #print("空")
    reposts_count=[None]*length
    uid=[None]*length
    bi_followers_count=[None]*length
    text=[None]*length
    user_description=[None]*length
    user_avatar=[None]*length
    id=[None]*length
    city=[None]*length
    friends_count=[None]*length
    province=[None]*length
    user_location=[None]*length
    followers_count=[None]*length
    verified_type=[None]*length
    picture=[None]*length
    statuses_count=[None]*length
    parent=[None]*length
    verified=[None]*length
    favourites_count=[None]*length
    username=[None]*length
    gender=[None]*length
    comments_count=[None]*length
    t=[None]*length
    count=0
    for index in range(ori_length):
        if fileJson[index]["text"]=='':
            count=count+1
            continue
        else:
            t[index-count] = fileJson[index]["t"]
            text[index-count] = fileJson[index]["text"]
            """
             reposts_count[index-count]=fileJson[index]["reposts_count"]
             uid[index-count] = fileJson[index]["uid"]
             bi_followers_count[index-count] = fileJson[index]["bi_followers_count"]
             user_description[index-count] = fileJson[index]["user_description"]
             user_avatar[index-count] = fileJson[index]["user_avatar"]
             id[index-count] = fileJson[index]["id"]
             city[index-count] = fileJson[index]["city"]
             friends_count[index-count] = fileJson[index]["friends_count"]
             province[index-count] = fileJson[index]["province"]
             user_location[index-count] = fileJson[index]["user_location"]
             followers_count[index-count] = fileJson[index]["followers_count"]
             verified_type[index-count] = fileJson[index]["verified_type"]
             picture[index-count] = fileJson[index]["picture"]
             statuses_count[index-count] = fileJson[index]["statuses_count"]
             parent[index-count] = fileJson[index]["parent"]
             verified[index-count] = fileJson[index]["verified"]
             favourites_count[index-count] = fileJson[index]["favourites_count"]
             username[index-count] = fileJson[index]["username"]
             gender[index-count] = fileJson[index]["gender"]
             comments_count[index-count] = fileJson[index]["comments_count"]
             """
    #return (length,t,text,reposts_count,uid,bi_followers_count,user_description,user_avatar,id,city,province,friends_count,user_location,followers_count,verified_type,statuses_count,picture,parent,verified,favourites_count,username,gender,comments_count)
    return (length,t,text)

"""
def find_lcsubstr(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
            if m[i+1][j+1]>mmax:
                mmax=m[i+1][j+1]
                p=i
    return (p-mmax+1,p,mmax) #s1[p-mmax:p]是最长子串
"""

def output(result):
    inputN = 50
    L=max(result[1])-min(result[1])
    l=L/inputN
    k=0
    #lcsk=[0]*100
    count=[0]*100
    while 1:
        k=k+1
        N=L/l
        N=int(N)
        s1=[0]*N
        s2=[1]*N
        interval=[0]*(N+1)
        for step in range(N+1):
            interval[step]=min(result[1])+step*l
        for step in range(N):
            i=0
            for i in range(result[0]):
                if interval[step]<result[1][i]<interval[step+1]:
                    s1[step]=1
        for i in range(N):
            if s1[i]==1:
                count[k] = count[k] + 1
        #lcs=find_lcsubstr(s1,s2)
        #lcsk[k]=lcs[2]
        #if lcsk[k]<inputN and lcsk[k]>lcsk[k-1]:
        if count[k] < inputN and count[k] > count[k - 1]:
            l=l/2
        else:
            #print("间隔为%d-%d,共%d个间隔，每个间隔%d个时间单位,有微博的间隔有%d个" % (interval[0], interval[N], N, l,count[k]))
            n=0
            #words=['']*lcsk[k]
            #for x in range(lcsk[k]):
            words = [''] * count[k]
            x=0
            for y in range(N):
                if s1[y] == 1:
                    words[x] = []
                    for z in range(result[0]):
                        #print(interval[lcs[0]]+x*l ,result[1][y] ,interval[lcs[0]]+(x+1)*l)
                        #if interval[lcs[0]]+x*l < result[1][y] <interval[lcs[0]]+(x+1)*l:
                        if interval[0] + y * l <= result[1][z] <= interval[0] + (y + 1) * l:
                            #words[x]=words[x].append(result[2][y])
                            words[x].append(z)
                            #print(result[2][z])
                    x=x+1
            return (words)


# def get_corpus(filedir):
#     """
#     获得路径下的所有文本的list，每个文本按空格分为list，形式为[[],[],[],·····]
#     path：语料路径
#     return：corpus_list语料  files_list语料文件名称
#     """
#     corpus_list = []
#     files_list = os.listdir(filedir)
#     index=0
#     for cur_filename in files_list:
#         index=index+1
#         print("共有%d个文件，现在分析第%d个"%(int(len(files_list)),index))
#         #if index==4:
#         #   break
#         localpath = filedir + "/" + cur_filename
#         result = resolveJson(localpath)
#         corpus_list.append(result[2])
#     #with open('F:\论文汇总\谣言检测\数据\A16rumdect\corpus.txt', 'wb+') as fw:  # with方式不需要再进行close
#     with open('F:\论文汇总\谣言检测\数据\A16rumdect\corpus.txt', mode='w', encoding='utf-8') as fw:
#         for corpus_index in corpus_list:
#             for weibo in corpus_index:
#                 if weibo=="":
#                     continue
#                 else:
#                     fw.write(str(weibo).replace(' ','')+'\n')
#                     #fw.write("\n".encode('utf-8'))
#     #print("corpus_list:" + str(corpus_list))
#     """
#      corpus_list=[]
#      index=0
#      with open('F:\论文汇总\谣言检测\数据\A16rumdect\\real_corpus.txt', mode='w', encoding='utf-8') as fw:
#          for cur_corpus in corpus_list:
#              index=index+1
#              print("%d/3093836"%index)
#              cur_corpus = ' '.join(jieba.cut(cur_corpus))
#              real_corpus.append(cur_corpus)
#              fw.write(str(cur_corpus)+ '\n')
#              #print(corpus_list)
#      """
#     return corpus_list


# def get_least_numbers_big_data(alist, k):
#     max_heap = []
#     length = len(alist)
#     if k <= 0 or k > length:
#         alist.extend([0] * (k - len(alist)))
#         return alist
#     k = k - 1
#     for ele in alist:
#         #ele = -ele 求最小数时取反
#         if len(max_heap) <= k:
#             heapq.heappush(max_heap, ele)
#         else:
#             heapq.heappushpop(max_heap, ele)
#     return max_heap


def get_tfidf(sum,result,tfidf,maxindex):
    weiboInIntervals=output(result) #按间隔分微博
    num_interval = len(weiboInIntervals)
    max_tfidf=[]
    for i in range(num_interval):
        #print("list:",list(map(lambda x:x+sum,weiboInIntervals[i])))
        cur_tfidf=tfidf[list(map(lambda x:x+sum,weiboInIntervals[i]))]
        #weight = cur_tfidf.data.tolist() # 将tf-idf矩阵中数据抽取出来
        weight=cur_tfidf[:,maxindex].sum(axis=0)
        #print(weight)
        max_k=weight.data.tolist()
        #max_k = get_least_numbers_big_data(weight, 5000)
        max_tfidf.append(max_k[0])
    return max_tfidf


def NNinput(sum,result,filename,tfidf,maxindex,ydata):
    #with open('F:\论文汇总\谣言检测\数据\A16rumdect\dataset\\'+str(filename)+'words.txt', 'w+') as cur_file:
    with open('/data/biantian/data/tfidf_words.txt', 'a+') as cur_file:#写到一个文件，各事件未加上[]
    #with open('/data/biantian/data/tfidf_dataset/' + str(filename) + 'words.txt', 'w+') as cur_file:
        cur_file.write(str(ydata))
        cur_file.write("\n")
        max_tfidf = get_tfidf(sum,result,tfidf,maxindex)
        for i in range(len(max_tfidf)):
            cur_file.write(str(max_tfidf[i]))
            cur_file.write("\n")
        cur_file.write("###")
    return


filedir='/data/biantian/data/Weibo'
#corpus_list = get_corpus(filedir)
corpus_list = []
f = open(str("/data/biantian/data/real_corpus.txt"),encoding='utf-8')
cur_file = f.read().replace("\n","\\\\")
cur_file = cur_file.split("\\\\")
corpus_list.extend(cur_file)
f.close()
#print("corpus_list:"+str(len(corpus_list)))
#corpus_list长度为3093837
#corpus_list=corpus_list[:30]#做测试用
vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(
    vectorizer.fit_transform(corpus_list))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
# word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
#print("tfidf:",tfidf)
#print("tfidfdata:",tfidf.data)
#arr=tfidf.toarray()
#sumtfidf=arr.sum(axis=0)
sumtfidf=tfidf.sum(axis=0)
a=sumtfidf.tolist()
sumtfidf=a[0]
maxindex = list(map(sumtfidf.index, heapq.nlargest(5000, sumtfidf)))
#tfidf.shape[0]长度为3093837
# cur_tfidf = tfidf[[1,2,4]]
# # weight = cur_tfidf.data.tolist() # 将tf-idf矩阵中数据抽取出来
# weight = cur_tfidf[:, maxindex]
# mak=weight.toarray().tolist()
# print("w",weight.toarray().tolist())
filecount=0
sum=0
fy = open('/data/biantian/data/Weibo.txt', encoding='utf-8')
y = fy.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data = []
for cur_y in y:
    cur_y = cur_y.split(" ")
    new_y = cur_y[0].split("\t")
    if len(new_y) == 3:
        new1_y = new_y[1].split(":")
        y_data.append(new1_y[1])
for filename in os.listdir(filedir):
    filecount=filecount+1
    print("分析第%d个文件，共%d个"%(filecount,len(os.listdir(filedir))))
    path = filedir + '/' + filename
    result = resolveJson(path)
    #sum有3093835条微博
    NNinput(sum,result,filename,tfidf,maxindex,y_data[filecount-1])
    sum=sum+result[0]
    #print("sum:"+str(sum))
    #计算每个事件里每个时间间隔前5000大的tfidf值，换行写入一个文件中