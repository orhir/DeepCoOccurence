import numpy as np
from scipy.signal import fftconvolve


def body_per_bins(i, total_conv, w_spatial, co_matrix, x, input_index, vec_input_index, strides):
    """
    the function is a body of the loop that run on the size of a co-occurrence matrix
    :param i: iteration number
    :param total_conv: final variable that contains the result
    :param w_spatial: the spatial filter
    :param co_matrix: the co-occurrence matrix
    :param x: the input of the layer
    :param input_index: the quantized input for the bins of co-occurrence matrix
    :param vec_input_index: reshaped input_index
    :param strides: optionally can be implemented stride
    :return:
    """
    # shape of input data
    x_shape = np.shape(x)
    w_shape = np.shape(w_spatial)
    # fx, fy, fin, fout = w_spatial.get_shape().as_list()
    # take only the line with specific bin
    bin_row = co_matrix[i, :]

    cof_matrix_bin = np.reshape(np.take(bin_row, vec_input_index), x_shape)

    # temp_mask = np.tile(temp_mask, [1, 1, 1, fout])# num_output_channels])

    temp_input = np.multiply(x, cof_matrix_bin)
    temp_input = np.expand_dims(temp_input, axis=4)
    w_3d = np.reshape(w_spatial, [w_shape[0], w_shape[1], w_shape[2], 1, 1])
    strid = strides.get_shape().as_list()
    temp_mask = np.astype(np.equal(input_index, i), float)
    temp_mask = np.expand_dims(temp_mask, axis=4)

    "Following lines don't change temp_mask value"
    # filter_ones = np.ones([1, 1, 1, 1, 1])
    # temp_mask = np.nn.conv3d(temp_mask, filter_ones, strides=[1, strid[0], strid[0], 1, 1], padding="SAME") //TODO: Downsampaling instead od 3D convolution

    temp_result = fftconvolve(temp_input, w_3d, mode="SAME")
    # temp_result = np.reduce_sum(temp_result, axis=4)
    temp_result = np.multiply(temp_result[:, :, :, :, 0], temp_mask[:, :, :, :, 0])
    total_conv = total_conv + temp_result

    return i + 1, total_conv, w_spatial, co_matrix, x, input_index, vec_input_index, strides


if __name__ == '__main__':

    num_bins = 5

    for i in range(num_bins):
        _, total_conv, _, _, _, _, _, _ = \
            body_per_bins(
                [0, total_conv, w_spatial, co_matrix, layer_input, input_index, vec_input_index, temp_strides])
