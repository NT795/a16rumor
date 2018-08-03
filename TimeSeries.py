# -*- coding: utf-8 -*-
import json
import os
import math
import jieba
import heapq
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def resolveJson(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    length=len(fileJson)
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
    for index in range(length):
        reposts_count[index]=fileJson[index]["reposts_count"]
        uid[index] = fileJson[index]["uid"]
        bi_followers_count[index] = fileJson[index]["bi_followers_count"]
        text[index] = fileJson[index]["text"]
        user_description[index] = fileJson[index]["user_description"]
        user_avatar[index] = fileJson[index]["user_avatar"]
        id[index] = fileJson[index]["id"]
        city[index] = fileJson[index]["city"]
        friends_count[index] = fileJson[index]["friends_count"]
        province[index] = fileJson[index]["province"]
        user_location[index] = fileJson[index]["user_location"]
        followers_count[index] = fileJson[index]["followers_count"]
        verified_type[index] = fileJson[index]["verified_type"]
        picture[index] = fileJson[index]["picture"]
        statuses_count[index] = fileJson[index]["statuses_count"]
        parent[index] = fileJson[index]["parent"]
        verified[index] = fileJson[index]["verified"]
        favourites_count[index] = fileJson[index]["favourites_count"]
        username[index] = fileJson[index]["username"]
        gender[index] = fileJson[index]["gender"]
        comments_count[index] = fileJson[index]["comments_count"]
        t[index] = fileJson[index]["t"]
    return (length,t,text,reposts_count,uid,bi_followers_count,user_description,user_avatar,id,city,province,friends_count,user_location,followers_count,verified_type,statuses_count,picture,parent,verified,favourites_count,username,gender,comments_count)


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
            #print("间隔为%d-%d,共%d个间隔，每个间隔%d个时间单位"%(interval[lcs[0]],interval[lcs[1]],N,l))
            print("间隔为%d-%d,共%d个间隔，每个间隔%d个时间单位,有微博的间隔有%d个" % (interval[0], interval[N], N, l,count[k]))
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


def get_corpus(filedir):
    """
    获得路径下的所有文本的list，每个文本按空格分为list，形式为[[],[],[],·····]
    path：语料路径
    return：corpus_list语料  files_list语料文件名称
    """
    corpus_list = []
    files_list = os.listdir(filedir)
    index=0
    for cur_filename in files_list:
        index=index+1
        print("共有%d个文件，现在分析第%d个"%(int(len(files_list)),index))
        #if index==4:
        #   break
        localpath = filedir + "/" + cur_filename
        result = resolveJson(localpath)
        corpus_list.append(result[2])
    #with open('F:\论文汇总\谣言检测\数据\A16rumdect\corpus.txt', 'wb+') as fw:  # with方式不需要再进行close
    with open('F:\论文汇总\谣言检测\数据\A16rumdect\corpus.txt', mode='w', encoding='utf-8') as fw:
        for corpus_index in corpus_list:
            for weibo in corpus_index:
                if weibo=="":
                    continue
                else:
                    fw.write(str(weibo).replace(' ','')+'\n')
                    #fw.write("\n".encode('utf-8'))
    #print("corpus_list:" + str(corpus_list))
    return corpus_list


def get_least_numbers_big_data(alist, k):
    max_heap = []
    length = len(alist)
    if k <= 0 or k > length:
        alist.extend([0] * (k - len(alist)))
        return alist
    k = k - 1
    for ele in alist:
        #ele = -ele 求最小数时取反
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)
        else:
            heapq.heappushpop(max_heap, ele)
    return max_heap


def get_tfidf(result,real_corpus):
    """
    real_corpus=[]
    index=0
    with open('F:\论文汇总\谣言检测\数据\A16rumdect\\real_corpus.txt', mode='w', encoding='utf-8') as fw:
        for cur_corpus in corpus:
            index=index+1
            print("%d/3093836"%index)
            cur_corpus = ' '.join(jieba.cut(cur_corpus))
            real_corpus.append(cur_corpus)
            fw.write(str(cur_corpus)+ '\n')
            #print(real_corpus)
    """
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(real_corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    #word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weiboInIntervals=output(result) #按间隔分微博
    num_interval = len(weiboInIntervals)
    max_tfidf=[]
    for i in range(num_interval):
        cur_tfidf=tfidf[weiboInIntervals[i]]
        #for cur_weibo in weibono:
        #cur_tfidf.append(tfidf[cur_weibo])
        weight = cur_tfidf.data.tolist() # 将tf-idf矩阵中数据抽取出来
        #print("max_k:")
        # 取前5000大的值存入文件
        max_k = get_least_numbers_big_data(weight, 5000)
        #print(max_k)
        max_tfidf.append(max_k)
    #print("max_tfidf:")
    #print(max_tfidf)
    """
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print
        u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print
            word[j], weight[i][j]
    """
    return max_tfidf


def NNinput(result,filename,corpus_list):
    with open('F:\论文汇总\谣言检测\数据\A16rumdect\dataset\\'+str(filename)+'words.txt', 'w+') as cur_file:
        max_tfidf = get_tfidf(result,corpus_list)
        for i in range(len(max_tfidf)):
            cur_file.write(max_tfidf[i])
            cur_file.write("\n")
    return True


filedir='F:\论文汇总\谣言检测\数据\A16rumdect\Weibo'
#corpus_list = get_corpus(filedir)
corpus_list = []
f = open(str("F:\论文汇总\谣言检测\数据\A16rumdect\\real_corpus.txt"),encoding='utf-8')
#cur_file = f.read()
cur_file = f.read().replace("\n","\\")
cur_file = cur_file.split("\\")
corpus_list.extend(cur_file)
f.close()
#print("corpus_list:"+str(corpus_list))
for filename in os.listdir(filedir):
    path = filedir + '/' + filename
    result = resolveJson(path)
    NNinput(result,filename,corpus_list)
    #计算每个事件里每个时间间隔前5000大的tfidf值，换行写入一个文件中