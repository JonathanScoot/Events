import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import os


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_look_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_look_path)

    def load(self, label_lookup_path, uid_look_path):
        # 加载分类字符串n*******对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_look_path).readlines()
        uid_to_human = {}
        # 一行行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符号
            line = line.strip('\n')
            # 按照TAB键分割
            parsed_items = line.split('\t')
            # 获取分类编号
            uid = parsed_items[0]
            # 获取分类名称
            human_string = parsed_items[1]
            # 保存编号字符串与分类名称的关系
            uid_to_human[uid] = human_string

        # 加载分类字符串对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    for root, dirs, files in os.walk('/Users/wangjie/Desktop/testimage/'):
        for file in files:
            # 载入图片
            if not any(map(file.endswith, ['.jpg', '.jpeg'])):
                continue
            print(file)
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            prediction = np.squeeze(predictions)
            image_path = os.path.join(root, file)
            print(image_path)
            # img = Image.open(image_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            top_k = prediction.argsort()[-1:]
            node_lookup = NodeLookup()
            print(top_k)
            for node_id in top_k:
                print(node_id)
                human_string = node_lookup.id_to_string(node_id)
                score = prediction[node_id]
                print('%s (score = %.5f)' % (human_string, score))

            print()


tf.Session.close()
del tf.Session