import tensorflow as tf  #导入tf框架
import numpy as np  #导入np模块
from tensorflow.examples.tutorials.mnist import input_data #导入mnist数据集输入函数
from PIL import Image  #导入PIL图像处理包
import matplotlib  #导入mat图形处理工具库
import matplotlib.pyplot as plt  #导入mat图形绘图工具

#模型路径，也就是我们训练得到的那个pb文件的实际路径，最好写绝对路径
model_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\mnist.pb"
#导入测试数据
mnsit_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\" #mnsit数据集文件本地存放路径
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True) #导入本地mnsit数据集，独热模式，这是实际的训练和测试数据
x_test = mnist.test.images #获取mnist的测试数据集的图像数据，
x_labels = mnist.test.labels #获取mnist的测试数据集中的标签数据

#导入之前的训练模型
with tf.Graph().as_default():#当前默认图
    output_graph_def = tf.GraphDef()  #定义一个图对象
    with open(model_path, "rb") as f: #打开之前训练的模型
        output_graph_def.ParseFromString(f.read()) #读取之前训练的模型文件数据
        tf.import_graph_def(output_graph_def, name="") #加载模型文件数据，使用之前训练好的模型

    with tf.Session() as sess:#定义一个tf会话
        tf.global_variables_initializer().run()#初始化
        # x_test = x_test.reshape(1, 28 * 28)
        input_x = sess.graph.get_tensor_by_name("input/x_input:0") #这里使用名称读取我们训练好模型的输入变量
        output = sess.graph.get_tensor_by_name("output:0")#这里使用名称读取我们训练好模型的输出算子

        # 【1】下面是进行批量测试----------------------------------------------------------
        pre_num = sess.run(output, feed_dict={input_x: x_test})  # 导入测试数据集图像数据，大约1万张照片，利用训练好的模型预测结果
        # 结果批量测试的准确率
        correct_prediction = tf.equal(pre_num, tf.argmax(x_labels, 1, output_type='int32'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={input_x: x_test})
        print('批量测试正确率：{0}'.format(acc))  #测试正确率大约在91%，因为是mnsit是标准数据集
        #以上测试正确率算法与之前训练数据任务中使用的的一致
        #【2】下面是进行单张图片的测试-----------------------------------------------------
        testImage = x_test[0]#我们取测试数据集中的第一张图片，一共有10000多张，索引0代表第一张
        test_input = testImage.reshape(1, 28 * 28)#矩阵升维度（1代表一个测试数据、28*28代表像素展平）
        #因为在mnsit数据集中是784的展平数据，我们需要还原成原来的28*28像素，以便显示
        pre_num = sess.run(output, feed_dict={input_x: test_input})#利用训练好的模型预测结果
        print('单个模型预测结果为：',pre_num[0])
        #显示测试的图片
        testImage = testImage.reshape(28, 28) #把第一维度（那个1代表数据集的数据样本数量）去掉，并还原成28*28标准矩阵数据
        testImage=np.array(testImage * 255, dtype="int32") #把归一化数据乘以255，得到标准的RGB数据格式，并转换成整形
        fig = plt.figure()
        plt.imshow(testImage, cmap='binary')  # 显示图片
        plt.title("This is test image")
        plt.show()
        #我们保存测定的图片，以方便接下来从单个样本中进行测试
        testImage = Image.fromarray(testImage)
        testImage = testImage.convert('1')
        testImage.save("E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\image\\show.jpg")
        #以上使用PIL包中的Image模块进行图片保存
        #当然我们也可是使用OpenCV，matplotlib等开源图像工具包
        ## matplotlib.image.imsave('path/name.jpg', im) #matplotlib的图片保存函数格式
        sess.close()#关闭会话
##如果要测试数据集中的其他图片，只需要把testImage = x_test[0]中的索引序号改变就行了，也可以写一个循环完成批量测试，
#但是要根据自己计算机配置量力而行，避免死机

