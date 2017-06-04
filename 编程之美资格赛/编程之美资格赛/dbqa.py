import thulac
import word2vec
from levenshtein import *
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

gobal_n_time_steps=200
gobal_rebuild=True

class DataManager():

    
    def __init__(self):
        #self.train_path = r"data\BoP2017-DBQA.train.txt"
        self.train_path = r"data\test.train.txt"
        self.dev_path = r"data\BoP2017-DBQA.dev.txt"
        self.train_data_true,self.train_data_false=self.load(self.train_path)
        #self.dev_data=self.load(self.dev_path)

    def next_train_batch(self,batch_size):
        idxt = np.arange(0, len(self.train_data_true))
        idxf = np.arange(0, len(self.train_data_false))
        np.random.shuffle(idxt)
        np.random.shuffle(idxf)
        batch_size>min(len(idxt),len(idxf),batch_size)
        idxt = idxt[0:batch_size]
        idxf = idxf[0:batch_size]
        input_x1 = [self.train_data_true[i][0] for i in idxt]
        input_x1.extend([self.train_data_false[i][0] for i in idxf])
        input_x2 = [self.train_data_true[i][1] for i in idxt]
        input_x2.extend([self.train_data_false[i][1] for i in idxf])
        input_y = [self.train_data_true[i][2][0] for i in idxt]
        input_y.extend([self.train_data_false[i][2][0] for i in idxf])
        return np.asarray(input_x1), np.asarray(input_x2),np.asarray(input_y)


    def load(self,path):

        last = ""
        subpath=path.split('.')[0]+'.'+path.split('.')[1] 
        splited_path = subpath + ".splited"
        bin_path = subpath + ".bin"
        cluster_path = subpath + ".cluster"

        f1=open(path,'r',encoding='utf-8')

        if gobal_rebuild:
            f2=open(splited_path,'w+',encoding="utf8",errors="ignore")

        lines=f1.readlines()
        f1.close()

        qa_pairs=list()

        thu1 = thulac.thulac(seg_only=True)

        for line in lines:
            words = line.split('\t')
            question=thu1.cut(words[1],text=True)
            answer=thu1.cut(words[2],text=True)
            if words[0]=='1':
                label=1
            else:
                label=0
            qa_pairs.append([question,answer,label])

            if gobal_rebuild:
                if question!=last:
                    last=question
                    f2.write(question)
                f2.write(answer) 


        

        if gobal_rebuild:
            f2.close()
            word2vec.word2vec(splited_path, bin_path, size=100, verbose=True)
            word2vec.word2clusters(splited_path, cluster_path, 100, verbose=True)

        model = word2vec.load(bin_path)

        print("\nDone to make vector!")
        veclized_data_true=list()
        veclized_data_false=list()

        fill_data=np.zeros([100,])
        for qa in qa_pairs:
            veclized_question=list()
            for voca in qa[0].split():
                try:
                    veclized_question.append(model[voca])
                except:
                    veclized_question.append(fill_data)
            if len(veclized_question)<gobal_n_time_steps:
                for i in range(gobal_n_time_steps-len(veclized_question)):
                    veclized_question.append(fill_data)
            if len(veclized_question)>gobal_n_time_steps:
                veclized_question=veclized_question[0:gobal_n_time_steps]

            veclized_answer=list()
            for voca in qa[1].split():
                try:
                    veclized_answer.append(model[voca])
                except:
                    veclized_answer.append(fill_data)
            if len(veclized_answer)<gobal_n_time_steps:
                for i in range(gobal_n_time_steps-len(veclized_answer)):
                    veclized_answer.append(fill_data)
            if len(veclized_answer)>gobal_n_time_steps:
                veclized_answer=veclized_answer[0:gobal_n_time_steps]

            veclized_label=list()
            if qa[2]==1:
                veclized_label.append([0,1])
                veclized_data_true.append([veclized_question,veclized_answer,veclized_label])

            else:
                veclized_label.append([1,0])
                veclized_data_false.append([veclized_question,veclized_answer,veclized_label])
        return veclized_data_true,veclized_data_false



def run_rnn():
    n_input = 100
    n_hidden_recc = 64
    n_hidden_nn = 64
    n_class = 2
    n_time_step = gobal_n_time_steps
    learning_rate = 0.001
    epoch = 1000

    x10 = tf.placeholder(dtype=tf.float32,shape=[None,n_time_step,n_input])
    x1 = tf.unstack(x10,n_time_step,1)
    cell1 = rnn.BasicLSTMCell(n_hidden_recc)
    o1,s1 = rnn.static_rnn(cell1,x1,dtype=tf.float32)
    w1 = tf.Variable(tf.random_normal([n_hidden_recc,n_hidden_nn]),dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal([n_hidden_nn,]))
    y1 = tf.matmul(o1[-1],w1) + b1

    x20 = tf.placeholder(dtype=tf.float32,shape=[None,n_time_step,n_input])
    x2 = tf.unstack(x20,n_time_step,1)
    cell2 = rnn.BasicLSTMCell(n_hidden_recc,reuse=True)
    o2,s2 = rnn.static_rnn(cell2,x2,dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal([n_hidden_recc,n_hidden_nn]),dtype=tf.float32)
    b2 = tf.Variable(tf.random_normal([n_hidden_nn,]))
    y2 = tf.matmul(o1[-1],w2) + b2

    x3 = tf.concat([y1,y2],1)
    w3 = tf.Variable(tf.random_normal([2 * n_hidden_nn,n_class]),dtype=tf.float32)
    b3 = tf.Variable(tf.random_normal([n_class,]))
    y3 = tf.matmul(x3,w3) + b3

    y = tf.placeholder(dtype=tf.float32,shape=[None,n_class,])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y3,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(y3,1),tf.argmax(y,1))
    accuracy = tf.reduce_max(tf.cast(correct,dtype=tf.float32))

    init = tf.global_variables_initializer()
    
    data_gen=DataManager()

    with tf.Session() as sess:
        sess.run(init)
        for iter in range(1,epoch + 1):
            input_x1,input_x2,input_y = data_gen.next_train_batch(10)
            sess.run(optimizer,feed_dict={x10:input_x1,x20:input_x2,y:input_y})
            acc = sess.run(accuracy,feed_dict={x10:input_x1,x20:input_x2,y:input_y})
            print("Iter %d of %d, the accuray is %.5f" % (iter,epoch,acc))
 

if __name__ == '__main__':
    run_rnn()
