import thulac
import word2vec
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

def encode_words(qset):
    thu1=thulac.thulac(seg_only=True)
    with open('tmp/splited_train_doc.txt','w+',encoding="utf8",errors="ignore") as f:
        for qa in qset:
            text=thu1.cut(qa[0][1],text=True)
            f.write(text)
            for a in qa:
                text=thu1.cut(a[2],text=True)
                f.write(text)

    #word2vec.word2phrase('tmp/splited_train_doc.txt', 'tmp/splited_train_doc_phrases.txt', verbose=True)
    word2vec.word2vec('tmp/splited_train_doc.txt', 'tmp/splited_train_doc.bin', size=100, verbose=True)
    word2vec.word2clusters('tmp/splited_train_doc.txt', 'tmp/splited_train_doc_clusters.txt', 100, verbose=True)

    pass

def dowork():
    qset=read_data()
    encode_words(qset)
    pass

if __name__=='__main__':
    #dowork()
    model = word2vec.load('tmp/splited_train_doc.bin')
    indexs=model.cosine(u'1951年')
    for index in indexs[0]:
        print(model.vocab[index])
        
        
