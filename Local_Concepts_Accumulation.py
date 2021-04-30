import tensorflow as tf


def lca(base_model_output):
    """
    Implementation of Local-Concepts Accumulation layer https://arxiv.org/pdf/2001.03992.pdf
    Input is the output of a base model.
    Output is tensor with the shape (None, amount of filters in base model output)"""
    # In original paper, authors use all kernel sizes more than (1, 1)
    # I'm not sure how they scramble matriсes into vector(None, FMs), so I used a MaxPool
    all_kernels = [(a, b) for a in range(1, base_model_output.shape[1] + 1, 1) for b in
                   range(1, base_model_output.shape[2] + 1, 1)]
    all_kernels.remove((1, 1))
    dense = tf.keras.layers.Dense(base_model_output.shape[-1], activation='relu', name='dense_lca')
    pools = []
    for i in range(len(all_kernels)):
        locals()['avg_pool' + str(i)] = tf.keras.layers.AveragePooling2D(pool_size=all_kernels[i], strides=None,
                                                                         padding="valid")(base_model_output)
        pools.append(dense(tf.keras.layers.Flatten(name=f'flatten_lca_{i}')(tf.keras.layers.MaxPool2D(
            pool_size=(locals()['avg_pool' + str(i)].shape[1],
                       locals()['avg_pool' + str(i)].shape[2]),
            strides=None, padding="valid", name=f'max_pool_lca_{i}')(locals()['avg_pool' + str(i)]))))

    output = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0), name='lambda_lca_mean')(pools)
    return output