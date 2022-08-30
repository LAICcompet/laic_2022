#coding:utf-8
import json 
import os
import argparse
from paddlenlp import Taskflow


def predict():
    ie = Taskflow(task='information_extraction', schema=[], task_path='./checkpoint/model_best', device_id=args.device)
    with open(args.output_file, 'w', encoding='utf8') as fw:
        with open(args.input_file, encoding='utf8') as fr:
            for lin in fr:
                lin_js = json.loads(lin)
                prompt, data = lin_js['prompt'], lin_js['content']
                ie.set_schema(prompt)
                if lin_js['result_list'][0]['text'] in ['正','负','一档','二档','三档']:
                    try:
                        result = ie(data)
                        p_lab = [{result[0][prompt][0]['text']}]
                    except:
                        p_lab = [{}]
                else:
                    try:
                        result = ie(data)
                        p_lab = [{start_end['start'], start_end['end']} for start_end in result[0][prompt]]
                    except:
                        p_lab = [{}]
                fw.write(json.dumps({'data':data, 'labels':str(p_lab)}, ensure_ascii=False)+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=1, help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--input_file", default="./input/test.json", help="Whether to run training.")
    parser.add_argument("--output_file", default="./result/output", help="Whether to run training.")
    args = parser.parse_args()

    predict()
