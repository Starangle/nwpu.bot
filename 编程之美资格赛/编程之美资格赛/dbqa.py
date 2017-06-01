from levenshtein import *
def read_data():
    path="data\BoP2017-DBQA.train.txt"
    last=""
    q=list()
    qset=list()
    for line in open(path,encoding='utf-8').readlines():
        words=line.split('\t')
        if last !=words[1]:
            if len(q)!=0:
                qset.append(q)
                q=[]
        q.append(words)
        last=words[1]
    return qset

if __name__=='__main__':
    read_data()
