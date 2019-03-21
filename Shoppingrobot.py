import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import os


def identify():

    lines = tf.gfile.GFile('/Users/wangjie/Documents/技术资料/机器学习/Tensorflow/modules/shengsai/shengsai2_lables.txt').readlines()
    uid_to_human = {}

    for uid, line in enumerate(lines):
        #去掉换行符号
        line = line.split('\n')
        uid_to_human[uid] = line

    def id_to_string(node_id):
        if node_id not in uid_to_human:
                return ''
        return uid_to_human[node_id]

    with tf.gfile.FastGFile('/Users/wangjie/Desktop/xiaosaimodule/output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for root, dirs, files in os.walk('/Users/wangjie/Desktop/xiaosaiimage'):
            for file in files:
                # 载入图片
                if not any(map(file.endswith, ['.jpg', '.jpeg'])):
                    continue
                print(file)
                image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                prediction = np.squeeze(predictions)
                image_path = os.path.join(root, file)
                #print(image_path)
                # img = Image.open(image_path)
                # plt.imshow(img)
                # plt.axis('off')
                # plt.show()
                top_k = prediction.argsort()[::-12]

                print(top_k[0])
                for node_id in top_k:
                    #print(node_id)
                    human_string = id_to_string(node_id)
                    score = prediction[node_id]
                    print('%s (score = %.5f)' % (human_string, score))

                print()


identify()