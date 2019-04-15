#-- 实验环境  win10 64位 + Anaconda3（python3.7） +TensorFlow 1.13 CPU版本
#IDE：pycharm

import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import  numpy as np
import  os
import  matplotlib.pyplot as plt

#mnist数据集本地存放地址
mnist_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\"
#导入本地数据集
mnist = input_data.read_data_sets(mnist_local_path,one_hot=True)#开启独热模式，方便标签判断
# 定义模型
with tf.name_scope('input'):
    #定义训练集的图像数据和标签数据输入
    x = tf.placeholder(shape=[None,784],dtype=tf.float32,name='x_input')
    y_labels = tf.placeholder(shape=[None,10],dtype=tf.float32,name='y_input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.truncated_normal(shape=[784,300],mean=0.0,stddev=0.1),name='W1') #W1
        W2 = tf.Variable(tf.truncated_normal(shape=[300,200],mean=0.0,stddev=0.1),name='W2') #W2
        W3 = tf.Variable(tf.zeros(shape=[200,10]),name='W3') #W3
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.truncated_normal(shape=[300],mean=0.0,stddev=0.1),name='b1') #b1
        b2 = tf.Variable(tf.truncated_normal(shape=[200],mean=0.0,stddev=0.1),name='b2') #b2
        b3 = tf.Variable(tf.zeros(shape=[10]),name='b3') #b3
    with tf.name_scope('op'):
        h1 = tf.nn.relu(tf.add(tf.matmul(x,W1),b1),name='h1') #h1
        h2 = tf.nn.relu(tf.add(tf.matmul(h1,W2),b2),name='h2') #h2
        y = tf.nn.softmax(tf.add(tf.matmul(h2,W3),b3),name='h3_out') #y
with tf.name_scope('loss'):
    loss = tf.reduce_mean(-tf.reduce_sum(y_labels * tf.log(y))) #loss

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss) #train


#输出算子
pre_num = tf.argmax(y,1,output_type='int32',name='output') #这是真正的输出，返回行向量最大值索引号，也就是结果
#计算正确率的算子
correct_prediction = tf.equal(pre_num,tf.argmax(y_labels,1,output_type='int32'))
acc = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name='acc')

#初始化参数
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
cost_summary = [] #临时变量，为测试loss与ecoch迭代的关系
#开始训练
for epoch in range(100):
    batch_x,batch_lables = mnist.train.next_batch(500)  #每次取500个图像数据以及标签样本
    sess.run(train_step,feed_dict={x:batch_x,y_labels:batch_lables}) #传值，训练
    if epoch % 10 ==0: #当epoch到10整数时，使用测试集计算loss值并保存到临时变量
        total_cost = sess.run(loss,feed_dict={x:mnist.test.images,y_labels:mnist.test.labels})
        cost_summary.append({'epoch':epoch + 1,'cost':total_cost})
#测试正确率
accu =  sess.run(acc,feed_dict={x:mnist.test.images,y_labels:mnist.test.labels})
print('测试正确率为：',accu)
#以下代码显示迭代次数与loss的关系
#plt.plot(list(map(lambda x: x['epoch'],cost_summary)),list(map(lambda x:x['cost'],cost_summary)))
#plt.show()

#使用saver方法保存模型
tf.add_to_collection('pre_num',pre_num) #定义collection
tf.add_to_collection('acc',acc) #定义collection
saver = tf.train.Saver()
saver.save(sess,"E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\normal\\modelsave")




