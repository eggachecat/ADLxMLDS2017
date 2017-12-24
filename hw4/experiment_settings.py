import tensorflow as tf

simple_settings = {
    "generator": {
        "is_deconv": True,
        "convs": [{
            "conv": {
                "num_outputs": 1,
                "kernel_size": 3,
                "stride": [3, 3],
                "activation_fn": tf.nn.relu,
            }
        }]
    },
    "discriminator": {
        "is_deconv": False,
        "convs": [{
            "conv": {
                "num_outputs": 32,
                "kernel_size": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            },
            "pool": {
                "kernel_size": 1,
                "stride": 1
            }
        }, {
            "conv": {
                "num_outputs": 32,
                "kernel_size": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            },
            "pool": {
                "kernel_size": 1,
                "stride": 1
            }
        }, {
            "conv": {
                "num_outputs": 32,
                "kernel_size": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            },
            "pool": {
                "kernel_size": 1,
                "stride": 1
            }
        }],
        "dense": [{
            "num_outputs": 2,
            "activation_fn": tf.nn.softmax,
        }]
    },
}
