#! -*- coding: utf-8 -*-
import numpy as np
from bert4keras.snippets import sequence_padding, DataGenerator
from models import build_tokenizer
from config import BaseConfig
import json


tokenizer = build_tokenizer(BaseConfig)
maxlen = BaseConfig.maxlen
categories = BaseConfig.categories


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def package_result(id, context, prediction):
    entities = {}
    for entity in prediction:
        start, end, label = entity
        if label in entities:
            entities[label].append((start, end))
        else:
            entities[label] = [(start, end)]
    entities_text = []
    for label in entities:
        entity_text ={"label": label, "span": entities[label]}
        texts = []
        for entity in entities[label]:
            start, end = entity
            text = context[start:end]
            texts.append(text)
        entity_text["text"] = texts
        entities_text.append(entity_text)
    entities = {}
    entities["context"] = context
    entities["id"] = id
    entities["entities"] = entities_text
    return entities


def save2file(filename, prediction):
    with open(filename, "w", encoding="utf-8") as fw:
        for entity in prediction:
            fw.write(json.dumps(entity, ensure_ascii=False))
            fw.write("\n")


