
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import request
import re
from PIL import Image
import matplotlib.pyplot as plt

"""
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))


"""

"""
x_data = np.random.rand(100)  # 设置一百个随机点
y_data = x_data*0.1 + 0.2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)  # 0.2的学习率
# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()            # 变量初始化

with tf.Session() as sess:
    sess.run(init)
    for step in range(800):
        sess.run(train)
        if step %20 == 0:
            print(step, sess.run([k, b]))
"""

"""
# 使用numpy生成两百个随机点
# 非线性回归问题
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]    # 范围内均匀分布的点
noise = np.random.normal(0, 0.02, x_data.shape)        # 随机生成两百个干扰项
y_data = np.square(x_data) + noise

# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)                           # 中间层的输出层 ，也是输出层的输入层

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
predicttion = tf.nn.tanh(Wx_plus_b_L2)

#
loss = tf.reduce_mean(tf.square(y - predicttion))

# 使用梯度下降法

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    predicttion_value = sess.run(predicttion, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predicttion_value, 'r-', lw=5)
    plt.show()
"""

"""
# 识别手写数据

from tensorflow.examples.tutorials.mnist import input_data


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 200
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))   # 返回一维张量中最大的值的位置

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter' + str(epoch) + ',Testing Accuracy' + str(acc))

"""

"""
# 卷积神经网络
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 200
# 计算一共有多少批次
n_batch = mnist.train.num_explames // batch_size

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 生成一个截断的正态分布
    return tf.Variable(initial)

# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第一个和四个参数默认设置为1


# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 改变X的格式转化为4D的向量
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 1表示1维图片(黑白) 3表示彩色图片

# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从一个平面抽取特征 1代表黑白图片
b_conv1 = bias_variable([32])    # 每个卷积核一个偏置值

# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 进行max_pooling

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5, 5, 32, 64])  # 5*5的采样窗口，32个卷积核从一个平面抽取特征
b_conv2 = bias_variable([64])    # 每个卷积核一个偏置值

# 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 进行max_pooling
"""

"""
# 模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/model/image/imagenet/inception-2015-12-05.tgz'



# 模型存放地址
inception_pretrain_model_dir = 'inception_model'
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 获取文件名及文件路径
"""
"""
filename = inception_pretrain_model_url.split('/')[1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print('Downloading', filename)
    r = request.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print('finish', filename)
"""

"""
# 解压缩文件
inception_pretrain_model_dir = 'inception_model'
filepath = '/Users/wangjie/PycharmProjects/Python project/inception_model'
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# 模型结构存放文件

log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pd为谷歌训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pd')
with tf.Session() as sess:
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir. sess.graph)
    writer.close()
"""

class Nodelookup(object):
    def __init__(self):
        label_lookup_path = '/Users/wangjie/PycharmProjects/Python project/inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = '/Users/wangjie/PycharmProjects/Python project/inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):  # 加载分类字符串n******  对应分类名称的文件夹
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readline()
        uid_to_human = {}
        # 一行行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符号
            line = line.strip('\n')
            # 按照'\t'进行分割
            parsed_items = line.split('\t')
            # 获取分类编号
            uid = parsed_items[0]
            # 获取分类名称
            human_string = parsed_items[1]
            # 保存编号字符串n*****与分类名称的字符关系
            uid_to_human[uid] = human_string

        # 加载分类字符串n*******对应分类编号1-10000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('   target_class:'):
                # 获取分类编号1-1000
                target_class =int(line.split(': ')[1])
            if line.startswith('   target_class_string'):
                # 获取编号字符串N*******
                target_class_string = line.split(' :')[1]
                # 保存分类编号1-1000对应分类名称的映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # 建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            # 建立分类编号1-10000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name


    # 传入分类编号1-10000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

# 创建一个图来存放Google训练好的模型


with tf.gfile.FastGFile('/Users/wangjie/PycharmProjects/Python project/inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 遍历目录
    for root, dirs, files in os.walk('/Users/wangjie/Desktop/testimage'):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = Nodelookup()
            for node_id in top_k:
                # 获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                # 获取可信度
                sorce = predictions[node_id]
                print('%s (sorce = %.5f)' % (human_string, sorce))

            print()














































