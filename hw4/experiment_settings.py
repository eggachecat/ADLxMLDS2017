import tensorflow as tf

simple_settings = {
    "generator": {
        "is_deconv": True,
        "convs": [{
            "conv": {
                "num_outputs": 256,
                "kernel_size": 3,
                "stride": [4, 4],
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 100,
                "kernel_size": 3,
                "stride": [4, 4],
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 3,
                "kernel_size": 3,
                "stride": [4, 4],
                "activation_fn": tf.nn.sigmoid,
            }
        }]
    },
    "discriminator": {
        "is_deconv": False,
        "convs": [{
            "conv": {
                "num_outputs": 32,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 32,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 32,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 32,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }],
        "dense": [{
            "num_outputs": 2,
            "activation_fn": tf.nn.softmax,
        }]
    },
}
