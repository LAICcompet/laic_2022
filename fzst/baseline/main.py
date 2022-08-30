'''
Author: xsun15 xsun15@mail.hfut.eu.cn
Date: 2022-05-30 17:30:34
LastEditors: xsun15 xsun15@mail.hfut.eu.cn
LastEditTime: 2022-05-31 10:33:15
FilePath: /baseline/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#! -*- coding: utf-8 -*-
from config import BaseConfig, load_data, load_json
import utils
from models import build_tokenizer,evaluate, Evaluator, build_model, get_model, NamedEntityRecognizer
from tqdm import tqdm
import sys
import os
import argparse



def train():

    samples, _ = load_json(BaseConfig.samples_path, True)
    samples = [i[1:] for i in samples]


    # 划分训练集和验证集
    train_size = int(len(samples) * BaseConfig.train_portion)
    valid_size = int((len(samples) - train_size) / 2)

    train_data = samples[0: train_size]
    valid_data = samples[train_size:train_size + valid_size]
    test_data = samples[train_size + valid_size:]

    # 构建模型
    model = build_model(BaseConfig)
    ner = NamedEntityRecognizer()
    evaluator = Evaluator(ner, valid_data, test_data,  model, utils.tokenizer)
    train_generator = utils.data_generator(train_data, BaseConfig.batch_size)
    # 模型训练
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=BaseConfig.epochs,
        callbacks=[evaluator]
    )

class Predict:

    def __init__(self):
        self.ner = NamedEntityRecognizer()
        self.model = get_model(BaseConfig)

    def predict(self):

        test, _ = load_json(BaseConfig.test)
        predictions = []
        X= 0
        Y = 0
        Z = 0
        for d in tqdm(test, ncols=100):
            R = self.ner.recognize(d[1], self.model, utils.tokenizer)
            R = set([(i[0],i[1]+1,i[2]) for i in R])
            T = set([tuple(i) for i in d[2:]])
            X += len(R & T)
            Y += len(set(R))
            Z += len(T)
            R = utils.package_result(d[0],d[1], R)
            predictions.append(R)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        utils.save2file(BaseConfig.output_file, predictions)
        return f1, precision, recall


if __name__=="__main__":
   
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=False, help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, help="Whether to run predictions on the test set.")
    parser.add_argument('--device', default="4", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--input_file", default="./input/test.json", help="Whether to run training.")
    parser.add_argument("--output_file", default="./result/output", help="Whether to run training.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.do_train:
        train()
    elif args.do_predict:
        f1, precision, recall = Predict().predict()
        print(f1, precision, recall)

