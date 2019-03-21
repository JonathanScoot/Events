import tensorflow as tf

import numpy as np

from PIL import Image



readImage=Image.open("/Users/wangjie/Desktop/cropped_panda.jpg")

readImage.show()



A=tf.truncated_normal([3,3,3,3],stddev=1.0)

matrix=np.asarray(readImage).astype("float32")



ma=tf.expand_dims(matrix,0)

#转换成tf变量，并转换成浮点类型32位

input_data=tf.Variable(ma,dtype="float32")

filter_data=tf.Variable(A,dtype="float32")

#进行卷积

conv2=tf.nn.conv2d(input_data,filter_data,[1,1,1,1],"SAME")

#进行池化

conv2=tf.nn.avg_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),

#删除一维度

res=tf.squeeze(conv2,axis=0)

#初始化

init_op=tf.global_variables_initializer()

with tf.Session as sess:

        sess.run(init_op)

        #执行

        a=res.eval()

       #防止图片乱码

       img2=Image.fromarray(a.astype(np.uint8))

       #显示图片

       img2.show()

