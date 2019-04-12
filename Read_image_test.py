#我们使用硬盘中现有的手写数字图像文件，也就是上一个测试环节保存的图片进行独立测试
#导入模块
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#模型路径，也就是我们训练得到的那个pb文件的实际路径，最好写绝对路径
model_path = "E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\mnist.pb"
#导入测试图片，上一个测试的输出图片，
#也可以自己写一个手写数字，但是请注意图片格式必须与其一致
#手写数字图片可以用很多图片处理工具进行预处理

#读取图片文件数据，这个图片是手写的0
testImage = Image.open("E:\\BaiduNetdiskDownload\\pycharm\\pycharm练习\\labdata\\image\\0.jpg")

#同上一次测试一样，导入训练好的数据模型
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # x_test = x_test.reshape(1, 28 * 28)
            input_x = sess.graph.get_tensor_by_name("input/x_input:0")
            output = sess.graph.get_tensor_by_name("output:0")
            # 对图片进行测试
            testImage = testImage.convert('L')
            testImage = testImage.resize((28, 28))
            test_input = np.array(testImage) #转换成np数据
            test_input = test_input.reshape(1, 28 * 28) #整形增加维度，以便符合模型参数
            pre_num = sess.run(output, feed_dict={input_x: test_input})  # 利用训练好的模型预测结果
            print('模型预测结果为：', pre_num[0])
            # 显示测试的图片
            fig = plt.figure()
            plt.imshow(testImage, cmap='binary')  # 显示原图片
            plt.title("This is test image ")
            plt.show()
            sess.close()
