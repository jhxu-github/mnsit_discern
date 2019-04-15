import  tensorflow as tf
import  numpy as np
from  tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt

#导入本地数据集
mnsit_local_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\mnist\\" #mnsit数据集文件本地存放路径
#导入本地mnsit数据集，独热模式，这是实际的训练和测试数据
mnist = input_data.read_data_sets(mnsit_local_path,one_hot=True)
#直接取得10000个测试图像样本数据和图像标签数据
x_test = mnist.test.images #直接把10000个测试数据样本的图片数据读成np数组
x_labels = mnist.test.labels # 直接把10000个测试数据样本的标签数据读成np数组

#恢复模型
sess = tf.Session() #需要一个会话
#本来我们需要重新构建整个graph，但是利用下面这个语句就可以加载整个graph了，方便
new_saver = tf.train.import_meta_graph("E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\normal\\modelsave.meta")
#加载模型中各种变量的值，注意这里不用文件的后缀
new_saver.restore(sess,"E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\normal\\modelsave")
#对应原模型的的add_to_collection()函数
pre_num = tf.get_collection('pre_num')[0]
acc = tf.get_collection('acc')[0]
graph = tf.get_default_graph() #图对象
x = graph.get_operation_by_name('input/x_input').outputs[0]#为了将placeholder加载出来
y_labels = graph.get_operation_by_name('input/y_input').outputs[0]#为了将placeholder加载出来
accu1 = sess.run(acc,feed_dict={x:x_test,y_labels:x_labels})
print("正确率：",accu1)

#测试单个图片
#读取图片文件数据，这个图片是手写的0
testImage = Image.open("E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\image\\0.jpg")
testImage = testImage.convert('L')  #换型
testImage = testImage.resize((28, 28)) #转换成28*28像素
test_input = np.array(testImage)  # 转换成np数据,并传递给输入变量
test_input = test_input.reshape(1, 28 * 28)  # 整形增加维度，以便符合模型参数
pre_num1 = sess.run(pre_num,feed_dict={x:test_input})
#这里可以看出，非归一化数据一样可以进行预测
print('模型预测结果为：', pre_num1[0])
# 显示测试的图片
plt.imshow(testImage, cmap='binary')  # 显示原图片,这里直接使用
plt.show()
sess.close()