#! -*- coding: utf-8 -*-


def get_param(meta_index, meta_id):

    channel_list = [32] * 2 + [40] * 4 + [80] * 4 + [96] * 4 + [192] * 4 + [320]
    stride_list = [2] + [1] + [2] + [1] * 3 + [2] + [1] * 7 + [2] + [1] * 4
    expand_list = [3, 3, 3, 6, 6, 6]
    kernel_list = [3, 5, 7, 3, 5, 7]

    assert meta_id in range(6), "meta_id error!"
    channel = channel_list[meta_index]
    stride = stride_list[meta_index]
    expand = expand_list[meta_id]
    kernel = kernel_list[meta_id]
    return expand, channel, kernel, stride
