# -*- coding: utf-8 -*-
import json
import os
import math
import jieba

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
            p=i+1
    return (p-mmax,p,mmax) #s1[p-mmax:p]是最长子串

def output(result):
    L=max(result[1])-min(result[1])
    l=L/inputN
    k=0
    lcsk=[0]*100
    while 1:
        k=k+1
        N=L/l
        N=int(N)
        s1=[0]*N
        s2=[1]*N
        interval=[0]*(N+1)
        for step in range(N):
            interval[step]=min(result[1])+step*l
        for step in range(N):
            i=0
            for i in range(result[0]):
                if interval[step]<result[1][i]<interval[step+1]:
                    s1[step]=1
        lcs=find_lcsubstr(s1,s2)
        lcsk[k]=lcs[2]
        if lcsk[k]<inputN and lcsk[k]>lcsk[k-1]:
            l=l/2
        else:
            print("间隔为%d-%d,共%d个间隔，每个间隔%d个时间单位"%(interval[lcs[0]],interval[lcs[1]],N,l))
            n=0
            words=[None]*N
            for x in range(N):
                for y in range(result[0]):
                    if interval[lcs[0]]+x*l < result[1][y] <interval[lcs[0]]+(x+1)*l:
                        words[x]=words[x].append(result[2][y])
            return (words)


def get_tf(text):
    """
    计算tf值
    text:该词所在文档，此处需要对text文本进行分词
    return:dict word_tf(该文本出现的词的tf值)
    """
    #print("text1"+str(text))
    text=','.join(jieba.cut(text))
    #print("text2"+str(text))
    num_words = len(text)
    word_freq = {}  # 词频dict
    word_tf = {}  # 词的tf值dict
    for i in range(num_words):
        word_count = 1
        for j in range(num_words):
            if i != j and text[i] != " ":
                if text[i] == text[j]:
                    word_count += 1
                    text[j] = " "
        if text[i] != " ":
            # word_freq[text[i]] = word_count
            word_tf[text[i]] = float(word_count / num_words)
    return word_tf


def get_idf(word, corpus_list):
    """
    计算idf值
    word：要计算的词
    corpus_list:包含所有语料的list，一个文件为其中一个元素
    return:该词的idf值
    """
    num_corpus = len(corpus_list)
    count = 0
    for cur_corpus in corpus_list:
        if word in set(cur_corpus):#此处中文注意分词，不然不准确
            count += 1
    idf = math.log(float(num_corpus / (count + 1)))
    return idf


def get_tfidf(cur_corpus, corpus_list):
    """
    分文本计算tfidf值
    cur_corpus:当前文本
    corpus_list：所有文本的list
    """
    cur_word_tfidf = {}
    word_tf = get_tf(cur_corpus)
    for word in word_tf:
        tf = word_tf[word]
        idf = get_idf(word, corpus_list)
        cur_word_tfidf[word] = tf * idf
    return cur_word_tfidf


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


def NNinput(result,filedir,corpus_list):
    words=output(result)
    num_corpus = len(corpus_list)
    #print(num_corpus)
    print(corpus_list[0])
    #corpus_list=result[2]
    with open('F:\论文汇总\谣言检测\数据\A16rumdect\words.txt', 'w+') as cur_file:
        for i in range(num_corpus):
            word_tfidf = get_tfidf(corpus_list[i], corpus_list)
            print("word_tfidf:"+str(word_tfidf))
            for cur_word in word_tfidf:
                cur_file.write(cur_word + ":" + str(word_tfidf[cur_word]) + "\n")
                # cur_file.close()
                #print("cur_word:"+cur_word + ":" + str(word_tfidf[cur_word]))
    return True


filedir='F:\论文汇总\谣言检测\数据\A16rumdect\Weibo'
#corpus_list = get_corpus(filedir)
corpus_list = []
f = open(str("F:\论文汇总\谣言检测\数据\A16rumdect\corpus.txt"),encoding='utf-8')
#cur_file = f.read()
cur_file = f.read().replace("\n"," ")
cur_file = cur_file.split(" ")
corpus_list.extend(cur_file)
f.close()
print("corpus_list:"+str(corpus_list))
inputN=10
for filename in os.listdir(filedir):
    path = filedir + '/' + filename
    result = resolveJson(path)
    NNinput(result,filedir,corpus_list)
    #读取每个文件前5000大的tfidf值[,,,]，换行写入一个文件中