import  tensorflow as tf
import  numpy as np
from  tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt

#导入训练好的模型以及mnist数据集
mnsit_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\" #mnsit数据集文件本地存放路径
#导入本地mnsit数据集，独热模式，这是实际的训练和测试数据
mnist = input_data.read_data_sets(mnsit_local_path,one_hot=True)
#训练好的模型文件存放路径
model_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\Re_build_nn.pb"

x_test = mnist.test.images #直接把10000个测试数据样本的图片数据读成np数组
x_labels = mnist.test.labels # 直接把10000个测试数据样本的标签数据读成np数组

#导入之前的训练模型(重点知识）
with tf.Graph().as_default():
    #代表在当前默认图下进行操作
    output_graph_def = tf.GraphDef()  # 定义一个图对象为默认图对象
    with open(model_path,'rb') as f:#读取模型文件
        output_graph_def.ParseFromString(f.read()) #读取之前训练的模型文件数据
        tf.import_graph_def(output_graph_def, name="")  # 加载模型文件数据，使用之前训练好的模型
    with tf.Session() as sess:  # 定义一个普通的tf会话
        tf.global_variables_initializer().run()  # 初始化参数
        input_x = sess.graph.get_tensor_by_name("input/x_input:0")  # 这里使用名称读取我们训练好模型的输入变量
        output = sess.graph.get_tensor_by_name("output:0")  # 这里使用名称读取我们训练好模型的输出算子
        # 【1】下面是进行批量测试----------------------------------------------------------
        #这里直接就可以调用output了
        pre_num = sess.run(output, feed_dict={input_x: x_test})  # 导入测试数据集图像数据，大约1万张照片
        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre_num,tf.argmax(x_labels,1,output_type='int32')),dtype=tf.float32))
        acc = sess.run(accuracy)
        print('批量测试准确率为：',acc)

        ##【2】下面是进行单张图片的测试-----------------------------------------------------
        TestImg = x_test[0]    #选择测试集中的第一张照片
        Test_single_img = TestImg.reshape(1,784) #升维，与输入样本的维度一致，1代表一个样本
        pre_num1 = sess.run(output, feed_dict={input_x:Test_single_img})
        print("这张图片是：",pre_num1[0])
        #下面显示测试的图片
        TestImg = TestImg.reshape(28,28) #去掉第一个维度，并恢复成28*28像素
        TestImg = np.array(TestImg * 255,dtype='uint8') #归一化数据乘以255并弄成整形
        plt.imshow(TestImg,cmap='binary') #加载图片
        plt.show()
        #保存图片，使用Image模块
        #TestImg = Image.fromarray(TestImg)  #读取数组成Image对象
        #TestImg = TestImg.convert('1') #转换成1型
        #TestImg.save("E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\image\\show.jpg")
        # 以上使用PIL包中的Image模块进行图片保存
        # 当然我们也可是使用OpenCV，matplotlib等开源图像工具包
        ## matplotlib.image.imsave('path/name.jpg', im) #matplotlib的图片保存函数格式
        sess.close()  # 关闭会话
    ##如果要测试数据集中的其他图片，只需要把testImage = x_test[0]中的索引序号改变就行了，也可以写一个循环完成批量测试，
    # 但是要根据自己计算机配置量力而行，避免死机