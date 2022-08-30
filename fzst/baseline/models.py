#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
from config import BaseConfig
import utils

def build_tokenizer(config):
    tokenizer = Tokenizer(config.bert_dict_path, do_lower_case=True)
    return tokenizer


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    y表示的是n个输入组合在一起的四维array
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


def build_model(config):
    roformer = build_transformer_model(
        config.bert_config_path,
        config.bert_checkpoint_path,
        'roformer'
    )
    output = GlobalPointer(len(config.categories), 64)(roformer.output)
    model = Model(roformer.inputs, output)
    #AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    optimizer = Adam(BaseConfig.learning_rate)
    model.compile(
        loss=global_pointer_crossentropy,
        optimizer=optimizer,
        metrics=[global_pointer_f1_score]
    )
    return model


def get_model(config):
    model = build_model(config)
    model.load_weights(config.best_model_path)
    return model


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, model, tokenizer, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=BaseConfig.maxlen)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], BaseConfig.categories[l])
            )
        return entities


def predict(data, ner, model, tokenizer):
    result = []
    for d in tqdm(data, ncols=100):
        R = ner.recognize(d[0], model, tokenizer)
    return result


def evaluate(data, ner, model, tokenizer):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(ner.recognize(d[0], model, tokenizer))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall



def expand_true(y_true):
    # 扩展y_true的范围标签
    """
    输出样式
    {0: {'label_dict': {1:{'origin': [3, 6], 'app_location': [[1, 6]]}}},1:...}
    """
    sample_num, label_num, word_num, _ = y_true.shape
    true_dict = {}
    for i in range(sample_num):
        sub_dict = {"label_dict":{}}
        for j in range(label_num):
            if 1 in y_true[i,j,:,:]:
                label_dict = {"origin": [], "app_location": []}
                location = np.where(y_true[i,j,:,:] == 1)
                label_dict["origin"] = [int(location[0]),int(location[1])]
                min_location = int(max(location[0] - 2, 0)[0])
                max_location = min(int(location[1] +2), word_num-1)
                for min_l in range(min_location,int(location[0][0]+1)):
                    for min_h in range(int(location[1]),max_location+1):
                        label_dict["app_location"].append([min_l,min_h])
                sub_dict["label_dict"][j] = label_dict
        true_dict[i] = sub_dict
    return true_dict





class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, ner, valid_data, test_data, model, tokenizer):
        self.best_val_f1 = 0
        self.ner = ner
        self.valid_data = valid_data
        self.test_data = test_data
        self.model = model
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(self.valid_data, self.ner, self.model, self.tokenizer)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(BaseConfig.best_model_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(self.test_data, self.ner, self.model, self.tokenizer)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )

