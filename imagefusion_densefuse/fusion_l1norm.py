import tensorflow as tf
import numpy as np

def L1_norm(source_ens):
    result = []
    size = len(source_ens)
    narrys = ["" for x in range(size)]
    temp_abs = ["" for x in range(size)]
    _l1 = ["" for x in range(size)]
    l1 = ["" for x in range(size)]
    mask_sign = ["" for x in range(size)]
    array_MASK = ["" for x in range(size)]
    # caculate L1-norm
    mask_value = 0
    for x in range(0, size):
        narrys[x] = source_ens[x]
        temp_abs[x] = tf.abs(narrys[x])
        _l1[x] = tf.reduce_sum(temp_abs[x],3)
        _l1[x] = tf.reduce_sum(_l1[x], 0)
        l1[x] = _l1[x].eval()
        mask_value += l1[x]
    dimension = source_ens[0].shape

    # caculate the map for source images
    for x in range(0, size):
        mask_sign[x] = l1[x]/mask_value
        array_MASK[x] = mask_sign[x]

    for i in range(dimension[3]):
        temp_matrix = 0
        for x in range(0, size):
            temp_matrix += array_MASK[x]*narrys[x][0,:,:,i]
        result.append(temp_matrix)

    result = np.stack(result, axis=-1)

    resule_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return resule_tf


