#-- 实验环境  win10 64位 + Anaconda3（python3.7） +TensorFlow 1.13 CPU版本
#IDE：pycharm
import  tensorflow as tf
import  os
from tensorflow.python.framework import  graph_util #tf框架计算图处理模块
from tensorflow.examples.tutorials.mnist import  input_data  #tf关于mnist数据集输入处理模块

#导入本地mnist数据集
mnist_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\"
mnist = input_data.read_data_sets(mnist_local_path,one_hot=True) #开启独热模式，方便标签判断

###创建三层前馈全连接神经网络模型，并初始化权重参数，并使用scope命名

with tf.name_scope('input'):#定义输入图像数据占位符张量，第一层命令为input，目的是方便模型在其他py程序中调用
    x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='x_input') #输入图片数据占位符
    y_label = tf.placeholder(dtype=tf.float32,shape=[None,10],name='y_input') #输入图片标签占位符

with tf.name_scope('layer'):#模型参数层的参数定义及其初始化
    with tf.name_scope('Weights'):
        #权重子命名
        W1 = tf.Variable(tf.truncated_normal([784,300],mean=0.0,stddev=0.1),name='W1')#隐藏层1权重矩阵张量1
        W2 = tf.Variable(tf.truncated_normal([300,200],mean=0.0,stddev=0.1),name='W2')#隐藏层2权重矩阵张量2
        W3 = tf.Variable(tf.zeros([200,10]),name='W3') #隐藏层3权重矩阵张量3
    with tf.name_scope('Biases'):
        #偏移量子命名
        b1 = tf.Variable(tf.truncated_normal(shape=[300],mean=0.0,stddev=0.1),name='b1')#隐藏层1偏移张量1
        b2 = tf.Variable(tf.truncated_normal(shape=[200],mean=0.0,stddev=0.1),name='b2')#隐藏层1偏移张量2
        b3 = tf.Variable(tf.zeros(shape=[10]), name='b3')  # 隐藏层1偏移张量2
    with tf.name_scope('W_p_b'):
        #模型计算算子命名
        h1 = tf.nn.relu(tf.add(tf.matmul(x,W1), b1),name='h1') #隐藏层1输出算子，使用relu激活函数
        h2 = tf.nn.relu(tf.add(tf.matmul(h1,W2),b2),name='h2') #隐藏层2输出算子，使用relu激活函数
        y = tf.nn.softmax(tf.add(tf.matmul(h2,W3),b3),name='final_result')
        #隐藏3是输出算子，也就是最终的输出，使用softm激活函数
with tf.name_scope('loss'):
    #定义代价函数算子
    loss =tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y))) #把计算得到到y和真实的y_label进行交叉熵计算
    #因为要考虑batch的输入，所以我们进行交叉熵计算后得进行mean平均计算在进行反向梯度传播计算
with tf.name_scope('train_step'):
    # 训练优化器算子
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    # 使用自适应梯度下降迭代算法求解代价函数，初始学习率0.01，效果相对好点
# 初始化 训练环境
sess = tf.InteractiveSession() #灵活的会话,运行算子运行run
#初始化参数
init = tf.global_variables_initializer()
sess.run(init)

###训练开始
for epoch in range(100):
    batch_x_images,batch_y_labels = mnist.train.next_batch(500) #获取训练数据集的图像数据和标签,一次500个样本
    #这里不能使用mnist.train.images.next_batch(500)和mnist.train.labels.next_batch(500)，nd数据没有这个方法
    train_step.run(feed_dict={x:batch_x_images,y_label:batch_y_labels}) #运行迭代算子
    #if epoch == 99:  测试单个输出结果
    #    y1 = y.eval(feed_dict={x:batch_x_images,y_label:batch_y_labels})
    #    print(y1[0])
#训练完后，可以自动使用模型参数进行输出预测计算
pre_num = tf.argmax(y,1,output_type='int32',name='output') #这是真正的输出，返回将是单行矩阵形式
# 必须给一个名字，因为在其他py程序中调用必须使用name进行调用
###测试准确率，也就是把计算得到的索引号与真实标签的索引号进行比较，统计相同数量，就得到了正确率
correct_prediction = tf.equal(pre_num,tf.argmax(y_label,1,output_type="int32")) #返回一个逐步对比的结果行向量，
# 相同为1，不相同为0，如果是单张图片，就只有一个值
##再对其求平均值，就可以计算到准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
#计算
a = accuracy.eval(feed_dict={x:mnist.test.images,y_label:mnist.test.labels}) #把测试集中所有的图片和标签数据用于测试
print("测试正确率：",a)

###输出保存模型及其参数单个pb文件
#形参output_node_names用于指定输出的节点名称,
#output_node_names=['output']对应前面定义的pre_num=tf.argmax(y,1,name="output"),最大值索引号计算算子
output_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['output'])
# convert_variables_to_constants函数，会将计算图中的变量取值以常量的形式保存
with tf.gfile.GFile('E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\Re_build_nn.pb', mode='wb') as f:
    # 需要指定模型文件的保存位置
    ###训练的目的就是为了获得模型及其参数数据###
    ##’wb’中w代表写文件，b代表将数据以二进制方式写入文件
    f.write(output_graph_def.SerializeToString())#以串行的方式写入

###完成此实验，会在指定的文件中看到一个pb文件，这个文件存放了我们的训练好的模型参数
