import jieba
import os
import re
import random


def rid_of_ad(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站', '更多更新免费电子书请关注www.cr173.com', '新语丝电子文库']
    for ads in ad:
        content = content.replace(ads, '')
    return content


def preprocess(corpus):
    pattern = r'[^\u4E00-\u9FA5]'
    regex = re.compile(pattern)
    replacements = ["\t", "\n", "\u3000", "\u0020", "\u00A0", " "]
    corpus = rid_of_ad(corpus)
    corpus = re.sub(regex, "", corpus)
    for replacement in replacements:
        corpus = corpus.replace(replacement, "")
    return corpus


def ReadData(path):
    content = []
    names = os.listdir(path)
    FileNum = len(names)
    ParaLength = 500 # token数
    ParaNum = 1000 # 抽取的段落数
    SelectNum = (ParaNum//FileNum) + 1
    for name in names:
        NovelName = path + '\\' + name
        with open(NovelName, 'r', encoding= 'ANSI') as f:
            con = f.read()
            con = preprocess(con)
            con = jieba.lcut(con)
            selectPos = len(con)//SelectNum
            for i in range(SelectNum):
                SelectStart =  random.randint(selectPos*i, selectPos*(i+1))
                para = con[SelectStart:SelectStart+ParaLength]
                content.append((name,para))
        f.close()
    content = content[:ParaNum]
    return content


def Dataset(content):
    traindata, trainlabel = [], []
    testdata, testlabel = [], []

    random.shuffle(content) # 打乱数据集
    for i in range(int(len(content)*0.9)): # 90%的数据做训练集
        trainlabel.append(content[i][0])
        traindata.append(content[i][1])

    for i in range(int(len(content)*0.9), int(len(content))):  # 剩下10%做测试集
        testlabel.append(content[i][0])
        testdata.append(content[i][1])

    return traindata, trainlabel,testdata, testlabel