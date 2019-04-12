#-- 实验环境  win10 64位 + Anaconda3（python3.7） +TensorFlow 1.32 CPU版本
#IDE：pycharm
import os
from tensorflow.examples.tutorials.mnist import input_data  #导入mnsit数据集处理模块
import tensorflow as tf  #导入Google tf深度学习框架
from tensorflow.python.framework import  graph_util  #此模块用于操作张量图

#print('tensortflow:{0}'.format(tf.__version__))       #显示tf版本
mnsit_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\" #mnsit数据集文件本地存放路径
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True) #导入本地mnsit数据集，独热模式，这是实际的训练和测试数据

#创建模型 实验使用单层前馈神经网络结构，网络结构的选择要根据数据集的维度、需求精度以及训练主机的算力的综合情况
#选取合适的神经网络机构层级
with tf.name_scope('input'): #输入层"input"下创建内存占位符张量
    x = tf.placeholder(tf.float32,[None,784],name='x_input')#数据输入节点名：x_input #mnsit图片是28*28 展平后就是784
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input') #标签输入节点名:y_input #mnsit图片标签共10个
with tf.name_scope('layer'):#定义权重与偏移量
    with tf.name_scope('W'):#权重
        #初始化参数矩阵一般使用随机数、正态分布随机数等
        W = tf.Variable(tf.zeros([784, 10]), name='Weights') #权重矩阵张量

    with tf.name_scope('b'):#偏移量
        b = tf.Variable(tf.zeros([10]),name='biases') #偏移量张量

    with tf.name_scope('W_p_b'):#计算公式算子
        Wx_plus_b = tf.add(tf.matmul(x, W), b, name='Wx_plus_b') #矩阵乘法+偏移量的算子

    y = tf.nn.softmax(Wx_plus_b, name='final_result') #输出使用softmax作为激活函数激活算子，注意y 与前面y_的区别

# 定义损失（代价）函数和优化方法
with tf.name_scope('loss'):#损失（代价）函数算子
    loss = -tf.reduce_sum(y_ * tf.log(y)) #使用交叉熵作为代价（代价）函数算子
with tf.name_scope('train_step'):#训练优化器算子
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss) #使用梯度下降迭代算法求解代价函数，学习率0.01
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)#使用自适应梯度下降迭代算法求解代价函数，学习率0.01，效果比上面的好点
    #print(type(train_step)) #查看算子类型

# 初始化 训练环境
sess = tf.InteractiveSession()  #定义tf会话
init = tf.global_variables_initializer() #初始化算子
sess.run(init)  #会话执行初始化

# 训练环节
for step in range(100):#为了节约时间，只进行100次迭代
    batch_xs,batch_ys =mnist.train.next_batch(500)  #每次训练数据取100个图像数据和标签，mnist数据集一共有5万多个
    train_step.run({x: batch_xs, y_: batch_ys})  #把实际训练数据以及标签数据传值给

# 测试模型准确率
pre_num=tf.argmax(y,1,output_type='int32',name="output")#输出节点名：output，最大值索引号算子
#以上的tf.argmax返回矩阵每行最大元素所在的索引位置
correct_prediction = tf.equal(pre_num,tf.argmax(y_,1,output_type='int32')) #计算正确率的算子，比较输出值与对应标签的相同数即可
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #正确率平均值算子
a = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}) #用测试数据计算正确率
print('测试正确率：{0}'.format(a)) #输出正确率，仅使用单层神经网络就可以达到90%以上的准确率

## 保存训练好的模型
#形参output_node_names用于指定输出的节点名称,
#output_node_names=['output']对应前面定义的pre_num=tf.argmax(y,1,name="output"),最大值索引号算子
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['output'])
# convert_variables_to_constants函数，会将计算图中的变量取值以常量的形式保存
with tf.gfile.FastGFile('E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\mnist.pb', mode='wb') as f:
    #需要指定模型文件的保存位置
    ###训练的目的就是为了获得模型及其参数数据###
    ##’wb’中w代表写文件，b代表将数据以二进制方式写入文件
    f.write(output_graph_def.SerializeToString())  #以串行的方式写入

###完成此实验，会在指定的文件中看到一个pb文件，这个文件存放了我们的训练好的模型参数