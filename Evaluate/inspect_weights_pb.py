"""
Created by Jago at 2019/12/20

inspect weights from pb file

"""
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import IPython.display

# python -u g:/CameraScene/slim_scene/Evaluate/inspect_weights_pb.py xxx.pb

PB_FILE = "sky_seg_20210624_quant_32232_trans.pb"
if len(sys.argv) > 1:
    PB_FILE = sys.argv[1]

os.makedirs(os.path.basename(PB_FILE).replace(".pb", ""), exist_ok=True)


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    open("result.html","w").write(html)
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(weight_dict):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "MEAN", "STD"]]
    print("WEIGHT NAME | SHAPE | MIN | MAX | MEAN | STD")
    for name, data in weight_dict.items():
        alert = ""
        if data.min() == data.max():
            alert += "<span style='color:red'>*** dead?</span>"
        if np.abs(data.min()) > 1000 or np.abs(data.max()) > 1000:
            alert += "<span style='color:red'>*** Overflow?</span>"
        # Add row
        table.append([
            name + alert,
            str(data.shape),
            "{:+9.4f}".format(data.min()),
            "{:+10.4f}".format(data.max()),
            "{:+9.4f}".format(data.std()),
        ])
        print("{:10s} | {} | {:+9.4f} | {:+10.4f} | {:+9.4f} | {:+9.4f}".format(name, data.shape, data.min(),
                                                                                data.max(), data.mean(), data.std()))
    display_table(table)


def plot_weight_histograms(weight_dict):
    num_param = len(weight_dict)
    num_cols = 4
    num_rows = 8

    count = -1
    for idx, (name, data) in enumerate(weight_dict.items()):
        if math.floor(idx / (num_cols * num_rows)) > count:
            fig, ax = plt.subplots(num_rows, num_cols, gridspec_kw={"hspace": 1},
                                   figsize=(30, 3 * num_rows))
            count += 1

        index = idx % (num_rows * num_cols)
        ax[math.floor(index / num_cols), index % num_cols].set_title(name)
        _ = ax[math.floor(index / num_cols), index % num_cols].hist(data.flatten(), 50)
        if index == num_cols * num_rows - 1:
            fig.savefig('{}/{}.png'.format(os.path.basename(PB_FILE).replace(".pb", ""), count))

    # plt.show()

    # for idx, (name, data) in enumerate(weight_dict.items()):
    #     name = name.replace('/', '_')
    #     fig = plt.figure()
    #     plt.hist(data.flatten(), 50)
    #     fig.savefig('test/{}.png'.format(name), dpi=fig.dpi)


def main(pb_file):
    with tf.Graph().as_default():
        graph_def = tf.compat.v1.GraphDef()
        with open(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')
        graph = tf.compat.v1.get_default_graph()
        constant_ops = [op for op in graph.get_operations() if op.type == "Const" and "Pad" not in op.name]
        params = [param for param in constant_ops if param.name.find('shape') < 0]
        weights = {}
        with tf.compat.v1.Session(graph=graph) as sess:
            for param in params:
                name = param.name
                data = sess.run(graph.get_tensor_by_name("%s:0" % name))
                weights[name] = data

        display_weight_stats(weights)
        plot_weight_histograms(weights)


if __name__ == '__main__':
    main(PB_FILE)
    exit(0)
