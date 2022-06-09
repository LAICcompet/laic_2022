'''
Author: xsun15 xsun15@mail.hfut.eu.cn
Date: 2022-05-31 10:33:07
LastEditors: xsun15 xsun15@mail.hfut.eu.cn
LastEditTime: 2022-05-31 16:25:37
FilePath: /baseline/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#! -*- coding: utf-8 -*-

import os
import json
import yaml


def load_json(filename, flag = False):
    """加载数据
    单条格式：[id, text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    dataset = []
    categories = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            d = []
            d.append(l["id"])
            d.append(l["context"])
            for entity in l["entities"]:
                label = entity["label"]
                for span in entity["span"]:
                    span_list = span.split(";")
                    d.append([int(span_list[0]),int(span_list[1]) -
                             1 if flag else int(span_list[1]),label])
                    categories.add(label)
            dataset.append(d)
    return dataset, categories

class BaseConfig(object):
    # 训练数据配置
    with open("./config/config.yml", "r", encoding="utf-8") as fr:
         entity = yaml.load(fr.read())
    # bert配置
    bert_config_path = entity["pretrain"]["bert_config_path"]
    bert_checkpoint_path = entity["pretrain"]["bert_checkpoint_path"]
    bert_dict_path = entity["pretrain"]["bert_dict_path"]

    # 参数设置
    maxlen = entity["pretrain"]["maxlen"]
    epochs = entity["pretrain"]["epochs"]
    batch_size = entity["pretrain"]["batch_size"]
    learning_rate = entity["pretrain"]["learning_rate"]

    samples_path = entity["pretrain"]["samples_path"]
    test = entity["pretrain"]["test"]


    train_portion =entity["pretrain"]["train_portion"]

    categories = []
    label_path = entity["pretrain"]["label_path"]
    with open(label_path,"r") as f:
        for label in f:
            categories.append(label.replace("\n",''))
    
    categories = list(sorted(categories))


    # 最优模型保存
    best_model_path = entity["pretrain"]["best_model_path"]
    output_file = entity["pretrain"]["output_file"]


