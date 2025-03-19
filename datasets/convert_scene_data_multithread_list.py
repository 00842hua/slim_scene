#!/data/anaconda2/bin/python
#coding:utf-8
# 用于转换logo图片语料
# 语料格式
# file_dir/A/A1.jpg
# file_dir/B/A1.jpg
# ....

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf
import dataset_utils
import threading
import time
import logging
import codecs

_RANDOM_SEED = 0


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

'''
nohup python convert_scene_data_multithread_list.py --file_list=/data_old/ffwang/SZBWG/wenwu/imgs/list.txt \
--record_dir=/data_old/ffwang/SZBWG/wenwu/tf_list --skip_check=1 --record_type=path \
--percent_validation=2 --num_shards=4 --data_name=scene --image_min_size=10 --thread_num=1 > log.txt 2>&1 &
'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('file_list', None, 'the list of images, 1st column path, 2nd column label')
tf.app.flags.DEFINE_string('record_dir', None, 'the tfrecord file save dir')
tf.app.flags.DEFINE_integer('percent_validation', 15, 'percent validation')
tf.app.flags.DEFINE_integer('num_shards', 12, 'number shards')
tf.app.flags.DEFINE_string('data_name', None, 'data name')
tf.app.flags.DEFINE_integer('skip_check', 0, 'skip image format check')
tf.app.flags.DEFINE_string('record_type', 'content', 'tfrecord save type content/path')
tf.app.flags.DEFINE_integer('image_min_size', 100, 'image min size, will abandon image less less than this size ')
tf.app.flags.DEFINE_integer('thread_num', 4, 'thread number')
tf.app.flags.DEFINE_string('label_file', None, 'specify label file, in order to specify id sequence')


def _get_pic_list(directory, class_name):
    photo_filenames = []
    for filename in os.listdir(directory):
        sub_item = os.path.join(directory, filename)
        if os.path.isdir(sub_item):
            photo_filenames.extend(_get_pic_list(sub_item, class_name))
        elif not filename.find('_tmpp.jpg') >= 0 \
                and not filename.find('_bw.jpg') >= 0 \
                and filename.find(".db") < 0:
            photo_filenames.append((sub_item, class_name))
    return photo_filenames


def _get_filenames_and_classes(file_list):
    class_names = []
    class_names_set = set()
    image_lines = open(file_list).read().splitlines()
    photo_filenames = []
    for line in image_lines:
        sep = line.split()
        if len(sep) != 2 and len(sep) != 3:
            logging.warning("***************** len(sep) != 2 and len(sep) != 3 : %s", line)
            continue
        is_crop = "0"
        if len(sep) == 3:
            is_crop = sep[2]
        photo_filenames.append((sep[0], sep[1], is_crop))
        if sep[1] not in class_names_set:
            class_names_set.add(sep[1])
            class_names.append(sep[1])
    logging.warning("***************** len(photo_filenames): %s", len(photo_filenames))
    logging.warning("***************** len(class_names): %s", len(class_names))

    class_names_ret = sorted(class_names)

    if FLAGS.label_file:
        lines = codecs.open(FLAGS.label_file).read().splitlines()
        class_names_specified = []
        for line in lines:
            sep = line.split(":")
            if len(sep) == 2:
                class_names_specified.append(sep[1])
        logging.warning("***************** len(class_names_ret): %s    len(class_names_specified) :%s",
                        len(class_names_ret), len(class_names_specified))
        print(class_names_ret[0])
        print(sorted(class_names_specified)[0])

        for k in class_names_specified:
            if k not in class_names_ret:
                print("%s NOT in class_names_ret" % (k))
                #raise ValueError('label file error!')

        class_names_ret = class_names_specified

        # if sorted(class_names_specified) != class_names_ret:
        #     for i in range(len(class_names_ret)):
        #         if class_names_ret[i] != sorted(class_names_specified)[i]:
        #             print("i:%d  class_names_ret:%s  class_names_specified:%s" %
        #                   (i, class_names_ret[i], sorted(class_names_specified)[i]))
        #             break
        #     raise ValueError('label file error!')
        # else:
        #     class_names_ret = class_names_specified
    return photo_filenames, class_names_ret


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        FLAGS.data_name, split_name, shard_id, FLAGS.num_shards)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, num_shards, filenames, class_names_to_ids, dataset_dir, thread_id, thread_num):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
    # num_per_shard = 1

    for shard_id in range(num_shards):
        if shard_id % thread_num != thread_id:
            continue

    # for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

        img_count = 0
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
            print("%s[%02d] start shard %d/%d, %d to %d"
                  % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id, shard_id, num_shards, start_ndx, end_ndx))
            for i in range(start_ndx, end_ndx):
                if i % 1000 == 0:
                    print("%s[%02d] running shard %d/%d, %d/%d"
                          % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id, shard_id, num_shards,
                             i-start_ndx, end_ndx-start_ndx))

                filename, class_name, is_crop = filenames[i]
                class_id = class_names_to_ids[class_name]

                if FLAGS.record_type=='content':
                    #trans picture to jpg
                    if FLAGS.skip_check!=1:
                        ret, tmpfile, height, width = \
                            dataset_utils.get_jpg_image(filename, min_size=FLAGS.image_min_size)
                        image_data = tmpfile.getvalue()
                    else:
                        ret = 0
                        height = 0
                        width = 0
                        with open(filename) as f:
                            image_data = f.read()
                    img_type = 'jpg'

                    if ret == 0:
                        example = dataset_utils.image_to_tfexample(
                            image_data, img_type, height, width, class_id, file_name=filename, is_crop=is_crop)
                        tfrecord_writer.write(example.SerializeToString())
                        img_count += 1
                    elif ret == -99:
                        print('%s[%02d]:: get jpg image %s not count'
                              % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id, filenames[i]))
                    else:
                        print('%s[%02d]:: get jpg image from %s fail'
                              % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id, filenames[i]))
                elif FLAGS.record_type=='path':
                    height = 0
                    width = 0
                    img_type = 'jpg'
                    example = dataset_utils.file_to_tfexample(filename, img_type, height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())
                else:
                    print("error! record_type=%s" % FLAGS.record_type)
                    raise Exception('record_type error')
        print("%s[%02d] finish shard %d: %d"
              % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id, shard_id, img_count))
    print("%s[%02d] finish all shard" % (time.strftime('%Y/%m/%d %H:%M:%S'), thread_id))


def main(_):
    if not FLAGS.file_list:
        raise ValueError('need set file_list')

    if not FLAGS.record_dir:
        raise ValueError('need set record_dir')

    if not FLAGS.data_name:
        raise ValueError('need set data_name')

    if not tf.gfile.Exists(FLAGS.record_dir):
        tf.gfile.MakeDirs(FLAGS.record_dir)

    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.file_list)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # print("get class_names_to_ids:")
    # print(class_names_to_ids)
    print('get class num %d' % len(class_names_to_ids))

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    num_validation = int(len(photo_filenames) * FLAGS.percent_validation / 100)

    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # _convert_dataset('train', training_filenames, class_names_to_ids, FLAGS.record_dir)
    # if num_validation>0:
    #     _convert_dataset('validation', validation_filenames, class_names_to_ids, FLAGS.record_dir)
    thread_list = []
    for i in range(FLAGS.thread_num):
        sthread = threading.Thread(target=_convert_dataset,
                                   args=('train', FLAGS.num_shards, training_filenames, class_names_to_ids,
                                         FLAGS.record_dir, i, FLAGS.thread_num))
        sthread.setDaemon(True)
        sthread.start()
        thread_list.append(sthread)
    for i in range(FLAGS.thread_num):
        thread_list[i].join()

    if FLAGS.percent_validation > 0:
        thread_list = []
        for i in range(FLAGS.thread_num):
            sthread = threading.Thread(target=_convert_dataset,
                                       args=('validation', FLAGS.num_shards, validation_filenames, class_names_to_ids,
                                             FLAGS.record_dir, i, FLAGS.thread_num))
            sthread.setDaemon(True)
            sthread.start()
            thread_list.append(sthread)
        for i in range(FLAGS.thread_num):
            thread_list[i].join()

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, FLAGS.record_dir)

    print("train: %d, validation: %d" % (len(training_filenames), len(validation_filenames)))

if __name__ == '__main__':
    tf.app.run()


