#! -*- coding: utf-8 -*-
from config import BaseConfig,  load_json
import utils
from models import  get_model, NamedEntityRecognizer
from tqdm import tqdm
import argparse
class Predict:

    def __init__(self):
        self.ner = NamedEntityRecognizer()
        self.model = get_model(BaseConfig)

    def predict(self):
        test, _ = load_json(args.input_file)
        predictions = []
        # X= 0
        # Y = 0
        # Z = 0
        for d in tqdm(test, ncols=100):
            R = self.ner.recognize(d[1], self.model, utils.tokenizer)
            R = set([(i[0],i[1]+1,i[2]) for i in R])
            # T = set([tuple(i) for i in d[2:]])
            # X += len(R & T)
            # Y += len(set(R))
            # Z += len(T)
            R = utils.package_result(d[0],d[1], R)
            predictions.append(R)
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        utils.save2file(args.output_file, predictions)
        # return f1, precision, recall


if __name__=="__main__":
   
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="",required=True, help="Whether to run training.")
    parser.add_argument("--output_file", default="",required=True, help="Whether to run training.")
    args = parser.parse_args()
    Predict().predict()
    # f1, precision, recall = Predict().predict()
    # print(f1, precision, recall)

