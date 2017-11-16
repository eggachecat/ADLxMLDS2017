from seq2seq_models.basic_model import *
from seq2seq_models.attention_model import *

MODEL_MAP = {
    "attention": {
        "train": AttentionModel_Train,
        "infer": AttentionModel_Infer
    },
    "basic": {
        "train": BasicModel_Train,
        "infer": BasicModel_Infer
    }
}
