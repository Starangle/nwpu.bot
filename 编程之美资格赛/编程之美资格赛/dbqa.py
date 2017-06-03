import thulac
import word2vec
from levenshtein import *
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

gobal_n_time_steps=200

class DataManager():

    
    def __init__(self):
        self.train_path = "data\BoP2017-DBQA.train.txt"
        self.dev_path = "data\BoP2017-DBQA.dev.txt"
        self.train_data=load(self.train_path)
        self.dev_data=load(self.dev_path)

    def next_train_batch(self,batch_size):
        idx = numpy.arange(0, len(self.train_data))
        numpy.random.shuffle(idx)
        idx = idx[0:batch_size]
        input_x1 = [self.train_data[0] for i in idx]
        input_x2 = [self.train_data[1] for i in idx]
        input_y = [self.train_data[2] for i in idx]
        return numpy.array(input_x1), numpy.array(input_x2),numpy.array(input_y)


    def load(self,path):

        last = ""
        splited_path = path.split('.')[0] + ".splited"
        bin_path = path.split('.')[0] + ".bin"
        cluster_path = path.split('.')[0] + ".cluster"

        thu1 = thulac.thulac(seg_only=True)
        
        f1=open(path,encoding='utf-8')
        f2=open(splited_path,'w+',encoding="utf8",errors="ignore")

        lines=f1.readlines()
        f1.close()

        qa_pairs=list()

        for line in lines:
            words = line.split('\t')
            question=thu1.cut(words[1])
            answer=thu1.cut(words[2])
            if words[0]=='1':
                label=1
            else:
                label=0
            qa_pairs.append([question,answer,label])

            if question!=last:
                f2.write(question)
            f2.write(answer)


        f2.close()

        word2vec.word2vec(splited_path, bin_path, size=100, verbose=True)
        word2vec.word2clusters(splited_path, cluster_path, 100, verbose=True)

        model = word2vec.load(bin_path)

        veclized_data=list()

        for qa in qa_pairs:
            veclized_question=list()
            for voca in qa[0]:
                veclized_question.append(model[voca])
            if len(veclized_question)<gobal_n_time_steps:
                for i in range(gobal_n_time_steps-len(veclized_question)):
                    veclized_question.append(np.zeros([100,]))
            if len(veclized_question)>gobal_n_time_steps:
                veclized_question=veclized_question[0:gobal_n_time_steps]

            veclized_answer=list()
            for voca in qa[1]:
                veclized_answer.append(model[voca])
            if len(veclized_answer)<gobal_n_time_steps:
                for i in range(gobal_n_time_steps-len(veclized_answer)):
                    veclized_answer.append(np.zeros([100,]))
            if len(veclized_answer)>gobal_n_time_steps:
                veclized_answer=veclized_answer[0:gobal_n_time_steps]

            veclized_label=list()
            if qa[2]==1:
                veclized_label.append([0,1])
            else:
                veclized_label.append([1,0])
            
            veclized_data.append(veclized_question,veclized_answer,veclized_label)

        return veclized_data



def run_rnn():
    n_input = 100
    n_hidden_recc = 64
    n_hidden_nn = 64
    n_class = 2
    n_time_step = gobal_n_time_steps
    learning_rate = 0.001
    epoch = 1000

    x1 = tf.placeholder(dtype=tf.float32,shape=[None,n_time_step,n_input])
    x1 = tf.unstack(x1,n_time_step,1)
    cell1 = rnn.BasicLSTMCell(n_hidden_recc)
    o1,s1 = rnn.static_rnn(cell1,x1,dtype=tf.float32)
    w1 = tf.Variable(tf.random_normal([n_hidden_recc,n_hidden_nn]),dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal([n_hidden_nn,]))
    y1 = tf.matmul(o1[-1],w1) + b1

    x2 = tf.placeholder(dtype=tf.float32,shape=[None,n_time_step,n_input])
    x2 = tf.unstack(x1,n_time_step,1)
    cell2 = rnn.BasicLSTMCell(n_hidden_recc)
    o2,s2 = rnn.static_rnn(cell2,x2,dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal([n_hidden_recc,n_hidden_nn]),dtype=tf.float32)
    b2 = tf.Variable(tf.random_normal([n_hidden_nn,]))
    y2 = tf.matmul(o1[-1],w2) + b2

    x3 = tf.concat([y1,y2],1)
    w3 = tf.Variable(tf.random_normal([2 * n_hidden_nn,n_class]),dtype=tf.float32)
    b3 = tf.Variable(tf.random_normal([n_class,]))
    y3 = tf.matmul(x3,w3) + b3

    y = tf.placeholder(dtype=tf.float32,shape=[n_class,])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y3,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(y3,1),tf.argmax(y,1))
    accuracy = tf.reduce_max(tf.cast(correct,dtype=tf.float32))

    init = tf.global_variables_initializer()
    
    data_gen=DataManager()

    with tf.Session() as sess:
        sess.run(init)
        for iter in range(1,epoch + 1):
            input_x1,input_x2,input_y = data_gen.next_train_batch()
            sess.run(optimizer,feed_dict={x1:input_x1,x2:input_x2,y:input_y})
            acc = sess.run(accuracy,feed_dict={x1:input_x1,x2:input_x2,y:input_y})
            print("Iter %d of %d, the accuray is %.5f" % (iter,epoch,acc))
            
if __name__ == '__main__':
    run_rnn()
        
        
