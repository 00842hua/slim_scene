#-*- coding:utf-8 -*-

import tensorflow as tf
import sys
import keras.backend as K
import matplotlib.pyplot as plt
import cv2
import io
import selectivesearch
import matplotlib.patches as mpatches

sys.path.append("../")

from preprocessing import preprocessing_factory
import numpy as np
import scipy.misc

PY3FLAG = sys.version_info > (3,0)
if PY3FLAG:
    open_fn = open
else:
    import codecs
    open_fn = codecs.open

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('imagelist_path', None, 'image to predict')
tf.app.flags.DEFINE_string('pb_path', None, 'pb file')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')
tf.app.flags.DEFINE_integer('image_size', 224, 'image_size.')
tf.app.flags.DEFINE_string('preprocess_type', 'inception_common', 'preprocess type')
tf.app.flags.DEFINE_string('last_node_names',None,'last_node_names')
tf.app.flags.DEFINE_string('output_dir',None,'store the output data')

#二值化定位热力图,绘制矩形
def draw_rect_by_binary(image_path, heatmap):

    image = cv2.imread(image_path)

    ret, binary = cv2.threshold(heatmap, 150, 255, cv2.THRESH_BINARY)

    # 获取图像轮廓坐标，其中contours为坐标值，此处只检测外形轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # cv2.boundingRect()返回轮廓矩阵的坐标值，四个值为x, y, w, h， 其中x, y为左上角坐标，w,h为矩阵的宽和高
        boxes = [cv2.boundingRect(c) for c in contours]
        for box in boxes[-1:]:
            x, y, w, h = box
            x = int(x * (image.shape[1]/224))
            w = int(w * (image.shape[1]/224))
            y = int(y * (image.shape[0]/224))
            h = int(h * (image.shape[0]/224))
            origin_pic = cv2.rectangle(image, (x, y), (x + w, y + h), (153, 153, 0), 2)
        #origin_pic = cv2.resize(origin_pic,(img.shape[0],img.shape[1]))
        cv2.imwrite(rectangle_image, origin_pic)


#selective search绘制预选框
def draw_rect_by_selectivesearch(image_path, heatmap):

    image = cv2.imread(image_path)

    img_resize = cv2.resize(image,(224,224))

    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR)
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img_resize, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    #经过nms处理后留下的预选框
    keep = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 3000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 2.5 or h / w > 2.5:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(img)
    boxes_rect = []
    boxes_mean = []
    for x, y, w, h in candidates:
        #print(x, y, x+w, y+h)
        heatmap = np.array(heatmap)
        mean = np.mean(heatmap[y:(y+h),x:(x+w)])
        boxes_rect.append([x,y,x+w,y+h])
        boxes_mean.append(mean)
    #NMS
    keep = calculate_nms(boxes_rect,boxes_mean)

    for i in keep[0:1]:
        x1 = int(boxes_rect[i][0] * (image.shape[1]/224))
        y1 = int(boxes_rect[i][1] * (image.shape[0]/224))
        x2 = int(boxes_rect[i][2] * (image.shape[1]/224))
        y2 = int(boxes_rect[i][3] * (image.shape[0]/224))

        origin_pic = cv2.rectangle(image, (x1, y1), (x2, y2), (153, 153, 0), 2)

    cv2.imwrite(selective_rect_image, origin_pic)
    # plt.imshow(origin_pic)
    # plt.show()

def calculate_nms(boxes_rect,boxes_mean,thresh = 0.3):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(0,len(boxes_rect)):
        x1.append(boxes_rect[i][0])
        y1.append(boxes_rect[i][1])
        x2.append(boxes_rect[i][2])
        y2.append(boxes_rect[i][3])
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #print(areas)
    index_order = np.array(boxes_mean).argsort()[::-1]
    #print(index_order)
    keep = []
    # print(x1)
    # print(y1)
    # print(x2)
    # print(y2)
    while index_order.size > 0:
        i = index_order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[index_order[1:]])
        yy1 = np.maximum(y1[i], y1[index_order[1:]])
        xx2 = np.minimum(x2[i], x2[index_order[1:]])
        yy2 = np.minimum(y2[i], y2[index_order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[index_order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        index_order = index_order[inds + 1]
    return keep



if __name__ == "__main__":
    if not FLAGS.pb_path:
        raise ValueError('need set pb_path')

    if not FLAGS.imagelist_path:
        raise ValueError('need set imagelist_path')

    if not FLAGS.labels_file:
        raise ValueError('need set labels_file')

    if not FLAGS.output_node_names:
        raise ValueError('need set output_node_names')

    if not FLAGS.preprocess_type:
        raise ValueError('need set preprocess_type')

    if not FLAGS.last_node_names:
        raise ValueError('need set last_node_names')

    if not FLAGS.output_dir:
        raise ValueError('need set output_dir')

    graph_def = tf.GraphDef()

    with open(FLAGS.pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

    sess = tf.Session()

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)

    #image_data = tf.gfile.FastGFile(FLAGS.image_path, 'rb').read()
    output_dir = FLAGS.output_dir + '/'

    for image_path in io.open(FLAGS.imagelist_path,encoding='utf-8').read().splitlines():

        image_name = image_path.split("/")[-1]

        print(image_name)

        heat_image_gcam = output_dir + 'heatmap_gcam/' + image_name

        heat_image_cam = output_dir + 'heatmap_cam/' + image_name

        rectangle_image = output_dir + 'rectangle_image/' + image_name

        selective_rect_image = output_dir + 'selective_rect/' + image_name

        with open(image_path, 'rb') as f:
            image_data = f.read()

        image_data_tensor = tf.image.decode_jpeg(image_data)

        preprocessed = image_preprocessing_fn(image_data_tensor, FLAGS.image_size, FLAGS.image_size)

        preprocessed_reshape = tf.expand_dims(preprocessed, 0)

        preprocessed_value = sess.run(preprocessed_reshape)
        # scipy.misc.imsave('./PB/crop.jpg', preprocessed_value[0])

        input_tensor = sess.graph.get_tensor_by_name("input:0")

        #获取cam_conv (?, 7, 7, 1280)
        cam_conv_nodename = 'Mnasnet/embedding'
        cam_conv_tensor = sess.graph.get_tensor_by_name(cam_conv_nodename + ":0")
        #(?, 224, 224, 1280)
        cam_conv_tensor_resize = tf.image.resize_images(cam_conv_tensor,[224,224])


        #获取cam_gap
        cam_gap_nodename = 'Mnasnet/Logits/Dropout/Identity'
        cam_gap = sess.graph.get_tensor_by_name(cam_gap_nodename + ":0")

        #获取gap后全连接的weight和bias
        cam_fc_w_name = 'Mnasnet/Logits/Conv2d_1c_1x1/weights/read'
        cam_fc_w = sess.graph.get_tensor_by_name(cam_fc_w_name + ":0")
        print(cam_fc_w.shape)
        cam_fc_b_name = 'Mnasnet/Logits/Conv2d_1c_1x1/biases/read'
        cam_fc_b = sess.graph.get_tensor_by_name(cam_fc_b_name + ":0")
        print(cam_fc_b.shape)
        cam_fc_value =  tf.nn.bias_add(cam_fc_w,cam_fc_b)


        #最后的输出tensor
        feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")

        #倒数的conv_tensor
        last_feature_tensor = sess.graph.get_tensor_by_name(FLAGS.last_node_names + ":0")

        result = sess.run([feature_tensor], {"input:0": preprocessed_value})

        pred = np.argsort(result[0][0])

        scores = feature_tensor[:, pred[-1]]
        last_scores = last_feature_tensor
        grads = K.gradients(scores, last_scores)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        iterate = K.function([input_tensor], [pooled_grads, last_scores[0]])

        pooled_grads_value, conv_layer_output_value = iterate([preprocessed_value])
        # last_node_names:0层的shape为1280
        for i in range(1280):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

        #grad_cam heatmap

        heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图

        heatmap = np.maximum(heatmap, 0)

        heatmap /= np.max(heatmap)

        #获取cam heatmap

        CAM_conv_resize = sess.run([cam_conv_tensor_resize], {"input:0": preprocessed_value})

        CAM_fc = sess.run([cam_fc_value], {"input:0": preprocessed_value})

        # print(np.array(CAM_conv_resize).shape)
        #
        # print(np.array(CAM_conv_resize).reshape(-1, 1280).shape)
        #
        # print(np.squeeze(np.array(CAM_fc))[:, pred[-1]].shape)

        CAM_heatmap = np.matmul(np.array(CAM_conv_resize).reshape(-1, 1280), np.squeeze(np.array(CAM_fc))[:, pred[-1]])

        CAM_heatmap = np.reshape(CAM_heatmap, (224, 224))

        # print(CAM_heatmap.shape)

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print("image shape is :{} ,{}".format(img.shape[1], img.shape[0]))

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同

        #CAM_heatmap = cv2.resize(CAM_heatmap,((img.shape[1], img.shape[0])))

        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式

        CAM_heatmap = np.uint8(CAM_heatmap)

        #print(img.shape)
        #
        # print(heatmap.shape)

        #显示灰度热力图
        # plt.imshow(heatmap,cmap='gray')
        # plt.show()

        # 绘制预选框
        try:
            heatmap_resize = cv2.resize(heatmap,(224,224))
            draw_rect_by_selectivesearch(image_path, heatmap_resize)
        except:
            print('continue')

        #二值化绘制矩形
        draw_rect_by_binary(image_path, CAM_heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

        CAM_heatmap = cv2.applyColorMap(CAM_heatmap,cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子

        CAM_heatmap = cv2.resize(CAM_heatmap,(img.shape[1],img.shape[0]))

        cam_heatmap_img = CAM_heatmap * 0.4 + img

        cv2.imwrite(heat_image_gcam, superimposed_img)

        cv2.imwrite(heat_image_cam, cam_heatmap_img)

        #image = plt.imread(new_heat_image)
        # plt.imshow(image)
        # plt.show()


