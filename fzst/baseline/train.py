#! -*- coding: utf-8 -*-
from config import BaseConfig,load_json
import utils
from models import  Evaluator, build_model, NamedEntityRecognizer



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
if __name__=="__main__":
    train()