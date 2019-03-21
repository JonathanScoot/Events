
# ! /usr/bin/env python2
# -*- coding: utf-8 -*-

'''''
构造一个卷积神经网络来训练mnist：
输入层： 784个输入节点
两个卷积层（每个都具有一个卷积和Pooling操作）：
    卷积操作：步长为1，边距为0，filter: 5x5
    Pooling(池化): 采用maxpooing, 2x2矩阵作为模板
输出层： 10个输出节点
'''

import input_data
import tensorflow as tf
import argparse
import numpy as np
import os
import struct


# 定义初始化操作
def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# 定义卷积和池化操作
''''' 
卷积后的图像高宽计算公式： W2 = (W1 - Fw + 2P) / S + 1 
其中：Fw为filter的宽，P为周围补0的圈数，S是步幅 
'''


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


''''' 
(1) 数据加载 
'''
# mnist数据路径
mnist_data_path = "/home/fcx/share/test/deeplearning/mnist/mnist_data"
# 加载mnist数据
mnist_data = input_data.read_data_sets(mnist_data_path, one_hot=True)

''''' 
(2) 输入层，输入张量x定义 
'''
# 神经网络输入层变量x定义
with tf.name_scope('input_layer'):
    x = tf.placeholder("float", [None, 784])  # 可以存放n个784（28x28）的数组

''''' 
(3) 第一层卷积层 
'''
with tf.name_scope('conv_layer_1'):
    # 定义卷积操作的filter为5x5的矩阵，且输出32个feature map, 输入的图片的通道数为1，因为是灰度图像
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 先将图像数据进行维度的变化
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 28x28的单通道图像
    # 卷积操作
    ''''' 
    输出h_conv1维度为：[-1, 28, 28, 32], 之所以还是28x28是因为参数padding='SAME' 
    如果采用padding='VALID'则输出为24x24, 用公式W2 = (W1 - Fw + 2P) / S + 1，24 = (28 - 5 + 2 * 0) / 1 + 1 
    '''
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # 将卷积完的结果进行pooling操作
    # 输出h_pool1维度为：[-1, 14, 14, 32]
    h_pool1 = max_pool_2x2(h_conv1)

''''' 
(3) 第二层卷积层 
'''
with tf.name_scope('conv_layer_2'):
    # 定义卷积操作的map为5x5的矩阵，且输出64个feature map, 输入的图片的通道数为32
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 卷积操作
    # 输出h_conv2维度为：[-1, 14, 14, 64]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 将卷积完的结果进行pooling操作
    # 输出h_pool2维度为：[-1, 7, 7, 64]
    h_pool2 = max_pool_2x2(h_conv2)
# 到此为止一张28x28的图片就变成了64个7x7的矩阵

''''' 
(4) 全连接层定义（full connection layer） 
'''
with tf.name_scope('full_connection_layer'):
    # 卷积层2输出作为输入和隐藏层之间的权重矩阵W_fc1,偏置项b_fc1初始化
    # 定义隐藏层的节点数
    hide_neurons = 1024
    # W_fc1 = weight_variable([7*7*64, hide_neurons])
    # 计算卷积层2输出的tensor，变化为一维的大小
    h_pool2_shape = h_pool2.get_shape().as_list()  # 得到一个列表[batch, hight, width, channels]
    fc_input_size = h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3]  # hight * width * channels
    W_fc1 = weight_variable([fc_input_size, hide_neurons])
    b_fc1 = bias_variable([hide_neurons])
    # 将卷积层2的输出张量扁平化作为全连接神经网络的输入
    fc_x = tf.reshape(h_pool2, [-1, fc_input_size])
    # 全连接层中隐藏层的输出
    fc_h = tf.nn.relu(tf.matmul(fc_x, W_fc1) + b_fc1)

    # 为了减少过拟合，在隐藏层和输出层之间加人dropout操作。
    # 用来代表一个神经元的输出在dropout中保存不变的概率。
    # 在训练的过程启动dropout，在测试过程中关闭dropout
    keep_prob = tf.placeholder("float")
    drop_fc_h = tf.nn.dropout(fc_h, keep_prob)

    # 隐藏层到输出层
    W_fc2 = weight_variable([hide_neurons, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(drop_fc_h, W_fc2) + b_fc2)

''''' 
(5) 设置训练方法，及其他超参数 
'''
# 设置期待输出值
y_ = tf.placeholder("float", [None, 10])
# 设置损失函数为交叉嫡函数(negative log-likelihood函数)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 单步训练操作,使用梯度下降算法,学习速率：0.01，损失函数：cross_entropy
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 使用更复杂的ADAM优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 初始化
init = tf.initialize_all_variables()

# 定义saver用来保存训练好的模型参数
saver = tf.train.Saver()

# 定义检测正确率的方法
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 用向量y和y_中的最大值进行比较
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 对正确率求均值


def train_and_test():
    # 建立会话
    with tf.Session() as sess:
        ''''' 
        (6) 开始训练 
        '''
        # 执行初始化
        sess.run(init)

        # 开始训练10000次
        for i in range(10000):
            tran_sets_batch = mnist_data.train.next_batch(100)  # 每次取得100个样本
            sess.run(train_step, feed_dict={x: tran_sets_batch[0], y_: tran_sets_batch[1], keep_prob: 0.5})

            # 每100次训练完都检测下测试的正确率,只从5000个测试样本中简单抽取100个进行测试
            if i % 100 == 0:
                validation_sets_batch = mnist_data.validation.next_batch(100)
                cur_rate = sess.run(accuracy,
                                    feed_dict={x: validation_sets_batch[0], y_: validation_sets_batch[1], keep_prob: 1})
                print('epoch %d accuracy: %s' % (i, cur_rate))

                # 保存训练后的模型参数
        saver.save(sess, './save_model_data_conv/model.ckpt')

        ''''' 
        (7) 用训练好的模型测试10000个样本最终的准确度 
        '''
        # 这样会报Resource Exhausted 显存资源不足的错误
        # rate = sess.run(accuracy, feed_dict = {x: mnist_data.test.images, y_: mnist_data.test.labels, keep_prob: 1})
        # print('Mean Accuracy is %s' % rate)

        # 只好多次加载统计结果
        sum_rate = 0
        for i in range(100):
            test_sets_batch = mnist_data.test.next_batch(100)
            epoch_rate = sess.run(accuracy, feed_dict={x: test_sets_batch[0], y_: test_sets_batch[1], keep_prob: 1})
            sum_rate = sum_rate + epoch_rate
        print('Mean Accuracy is %s' % (sum_rate / 100))

    # 检验是否是bmp图片 54字节头 + 数据部分


def checkIsBmp(file):
    with open(file, 'rb') as f:
        head = struct.unpack('<ccIIIIIIHH', f.read(30))  # 将读取到的30个字节，转换为指定数据类型的数字
        # print(head)
        if head[0] == b'B' and head[1] == b'M':
            # print('%s 总大小：%d, 图片尺寸：%d X %d, 颜色数：%d' % (file, head[2], head[6], head[7], head[9]))
            return True, head[2], head[9]  # 返回图片总大小，以及一个像素用多少bit表示
        else:
            # print('%s 不是Bmp图片' % file)
            return False, 0, 0


def test_my_data():
    with tf.Session() as sess:
        ''''' 恢复训练好的数据 '''
        model_data = tf.train.latest_checkpoint('./save_model_data_conv/')
        saver.restore(sess, model_data)
        ''''' 
        #只好多次加载统计结果 
        sum_rate = 0 
        for i in range(100): 
            test_sets_batch = mnist_data.test.next_batch(100) 
            epoch_rate = sess.run(accuracy, feed_dict = {x: test_sets_batch[0], y_: test_sets_batch[1], keep_prob: 1}) 
            sum_rate = sum_rate + epoch_rate 
        print('The Accuracy tested by MNIST Test samples is: %s' % (sum_rate / 100)) 
        '''

        print('Start recognizing my image:')
        my_data_path = './my_test_images/'
        print(os.listdir(my_data_path))
        for image_name in os.listdir(my_data_path):
            image_path = os.path.join(my_data_path, image_name)
            ret, file_size, bitCnt_per_pix = checkIsBmp(image_path)
            if ret == True:
                with open(image_path, 'rb') as f:
                    formate = '%dB' % file_size
                    data_bytes = struct.unpack(formate, f.read(file_size))  # 按unsigned char 读取文件
                    image_np = np.zeros((1, 784))
                    # print(image_np.shape)
                    step = bitCnt_per_pix / 8
                    start_pos = 54  # 54字节的头信息
                    if bitCnt_per_pix == 8:  # 如果是8位深度的bmp图片，有调色板
                        start_pos = start_pos + 1024  # 1024字节的调色板

                    for i in range(784):
                        pos = start_pos + i * step
                        if bitCnt_per_pix == 8:
                            pix_value = data_bytes[pos]
                        elif bitCnt_per_pix == 24:
                            # gray = red * 0.299 + green * 0.587 + blue * 0.114 RGB和灰度图的转换
                            pix_value = data_bytes[pos] * 0.299 + data_bytes[pos + 1] * 0.587 + data_bytes[
                                pos + 2] * 0.114
                        image_np[0][i] = pix_value / 255.0  # [0..255] --> [0.0... 1.0]
                        # print(image_np[0][i])
                    max_idx, out = sess.run([tf.argmax(y, 1), y], feed_dict={x: image_np, keep_prob: 1})
                    print('####The image name: %s, predict number is: %d' % (image_name, max_idx))


def main():
    if ARGS.test:
        print('************** Predict by Neural network ***************')
        test_my_data()
    else:
        print('************** Train and Test accuracy of Neural network ***************')
        train_and_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--test',
        default=False,
        action='store_true',
        help='Excute training network or testing network.'
    )

    ARGS = parser.parse_args()
    print('ARGS: %s' % ARGS)
    tf.app.run()

''''' 
(1)tf.argmax(input, axis=None, name=None, dimension=None) 
此函数是对矩阵按行或列计算最大值 
参数 

    input：输入Tensor 
    axis：0表示按列，1表示按行 
    name：名称 
    dimension：和axis功能一样，默认axis取值优先。新加的字段 

返回：Tensor 行或列的最大值下标向量 

(2)tf.equal(a, b) 
此函数比较等维度的a, b矩阵相应位置的元素是否相等，相等返回True,否则为False 
返回：同维度的矩阵，元素值为True或False 

(3)tf.cast(x, dtype, name=None) 
将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以 

(4)tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None) 
 功能：求某维度的最大值 
(5)tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None) 
功能：求某维度的均值 

参数1--input_tensor:待求值的tensor。 
参数2--reduction_indices:在哪一维上求解。0表示按列，1表示按行 
参数（3）（4）可忽略 
例：x = [ 1, 2 
          3, 4] 
x = tf.constant([[1,2],[3,4]], "float") 
tf.reduce_mean(x) = 2.5 
tf.reduce_mean(x, 0) = [2, 3] 
tf.reduce_mean(x, 1) = [1.5, 3.5] 

(6)tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
从截断的正态分布中输出随机值 

    shape: 输出的张量的维度尺寸。 
    mean: 正态分布的均值。 
    stddev: 正态分布的标准差。 
    dtype: 输出的类型。 
    seed: 一个整数，当设置之后，每次生成的随机数都一样。 
    name: 操作的名字。 

（7）tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
从标准正态分布中输出随机值 

(8) tf.nn.conv2d(input, filter, strides, padding,  
                 use_cudnn_on_gpu=None, data_format=None, name=None) 
在给定的4D input与 filter下计算2D卷积 
    1，输入shape为 [batch, height, width, in_channels]: batch为图片数量，in_channels为图片通道数 
    2，第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width,  
        in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数， 
        卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input 
        的第四维 
    3，第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4 
    4，第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍） 
    5，第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true 

    结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。 

(9)tf.nn.max_pool(value, ksize, strides, padding, name=None) 
参数是四个，和卷积很类似： 
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map， 
    依然是[batch, height, width, channels]这样的shape 
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们 
    不想在batch和channels上做池化，所以这两个维度设为了1 
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1] 
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME' 
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式 

(10) tf.reshape(tensor, shape, name=None) 
函数的作用是将tensor变换为参数shape的形式。 
其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小， 
函数会自动计算，但列表中只能存在一个-1。（当然如果存在多个-1，就是一个存在多解的方程了） 

(11)tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)  
为了减少过拟合，随机扔掉一些神经元，这些神经元不参与权重的更新和运算 
参数： 
    x            :  输入tensor 
    keep_prob    :  float类型，每个元素被保留下来的概率 
    noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。 
    seed         : 整形变量，随机数种子。 
    name         : 名字，没啥用。  
'''