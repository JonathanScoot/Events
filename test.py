#===============================卷积神经网络训练手写识别===============================  
import tensorflow as tf  
import input_data  
mnist=input_data.read_data_sets('/data/mnist',one_hot=True)  
  
def compute_accuracy(v_xs,v_ys):  
    global prediction#申明全局变量  
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})  
    #判断是否等,argmax为最大概率的索引  
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))  
    #tf.cast为转化格式  
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})  
    return result  
  
def weight_variable(shape):  
    initial=tf.random_normal(shape=shape,stddev=0.1)  
    return tf.Variable(initial)  
  
def bias_variable(shape):  
    initial=tf.constant(0.1,shape=shape)  
    return tf.Variable(initial)  
  
def conv2d(x,W):  
    #strides等于采样步长,中间两个1表示xy方向步长都为1,第一个和最后一个恒为1  
    #SAME为抽取出来长宽与原图一样卷积时包含边界之外的像素,边界以外全0填充  
    #若padding为VALID则无边界之外的像素  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  
  
def max_pool_2x2(x):  
    #池化层过滤器边长为2移动步长为2,采用全0填充(SAME)  
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  
  
xs=tf.placeholder(tf.float32,[None,784])#784个像素点28*28  
ys=tf.placeholder(tf.float32,[None,10])#None为这个维度输入待定  
keep_prob=tf.placeholder(tf.float32)  
x_image=tf.reshape(xs,shape=[-1,28,28,1])#最后一个是色彩深度,-1与None一样  
  
#卷积+池化层1  
W_conv1=weight_variable([5,5,1,32])#patch 5*5输入深度为1,输出深度为32,即32个卷积核  
b_conv1=bias_variable([32])  
h_conv1=tf.nn.relu(tf.add(conv2d(x_image,W_conv1),b_conv1))#output:28*28*32  
h_pool1=max_pool_2x2(h_conv1)#output:14*14*32  
  
#卷积+池化层2  
W_conv2=weight_variable([5,5,32,64])#前一次输出深度为32因此这次输入深度为32，输出深度64  
b_conv2=bias_variable([64])  
h_conv2=tf.nn.relu(tf.add(conv2d(h_pool1,W_conv2),b_conv2))#output:14*14*64  
h_pool2=max_pool_2x2(h_conv2)#output:7*7*64  
  
#全连接隐层  
W_fc1=weight_variable([7*7*64,1024])  
b_fc1=bias_variable([1024])  
h_pool2_flat=tf.reshape(h_pool2,shape=[-1,7*7*64])  
h_fc1=tf.nn.relu(tf.add(tf.matmul(h_pool2_flat,W_fc1),b_fc1))  
f_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)  
  
#输出层  
W_fc2=weight_variable([1024,10])  
b_fc2=bias_variable([10])  
prediction=tf.nn.softmax(tf.add(tf.matmul(f_fc1_drop,W_fc2),b_fc2))  
  
#loss+train  
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),  
                                reduction_indices=[1]))#分类一般用交叉熵  
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
sess=tf.Session()  
sess.run(tf.global_variables_initializer())  
for i in range(1000):  
    batch_xs,batch_ys=mnist.train.next_batch(100)#每次学习100个d  
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.8})  
    if i%50==0:  
        print(compute_accuracy(mnist.test.images,mnist.test.labels))  
sess.close()  