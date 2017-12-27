import tensorflow as tf

simple_settings = {
    "generator": {
        "is_deconv": True,
        "optimizer": {
            "type": tf.train.AdamOptimizer,
            "parameters": {
                "learning_rate": 0.01
            }
        },
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
                "activation_fn": tf.nn.tanh,
            }
        }]
    },
    "discriminator": {
        "is_deconv": False,
        "optimizer": {
            "type": tf.train.AdamOptimizer,
            "parameters": {
                "learning_rate": 0.01
            }
        },
        "convs": [{
            "conv": {
                "num_outputs": 64,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 128,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 256,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }, {
            "conv": {
                "num_outputs": 512,
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "same",
                "activation_fn": tf.nn.relu,
            }
        }],
        "dense": [{
            "num_outputs": 1,
            "activation_fn": None  ,
        }]
    },
}
