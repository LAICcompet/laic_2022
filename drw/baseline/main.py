#coding:utf-8
import json 
import os
import argparse
from paddlenlp import Taskflow



def data_read(path):
    data_dir = {}
    for file_name in os.listdir(path):
        if file_name != '案件要素':
            data_dir[file_name] = []
            with open(os.path.join(path, file_name), encoding='utf8') as f:
                for line in f:
                    line_js = json.loads(line)
                    data_dir[file_name].append(line_js)
        else:
            for case_plot_name in os.listdir(os.path.join(path, file_name)):
                data_dir[case_plot_name] = []
                with open(os.path.join(path, os.path.join(file_name, case_plot_name)), encoding='utf8') as f:
                    for line in f:
                        line_js = json.loads(line)
                        data_dir[case_plot_name].append(line_js)
    return data_dir


def predict():
    data = data_read(args.input_file)
    output = open(args.output_file, 'w', encoding='utf8')
    ie = Taskflow(task='information_extraction', schema=[], task_path='小样本多任务/baseline/checkpoint/model_best', device_id=args.device)
    for task_name,task_data_list in data.items():
        if task_name != 'ner':
            tq_name = 'data'
            if task_name == '刑档':
                prompt = task_name + '[一档,二档,三档]'
            else:
                prompt = task_name + '[正,负]'
        else:
            tq_name = 'text' 
            prompt = ' '

        if prompt != ' ':  # 分类任务
            for line in task_data_list:
                ie.set_schema(prompt)
                try:
                    result = ie(line[tq_name].strip())
                    p_lab = [[result[0][prompt][0]['text']]]
                except:
                    p_lab = [[]]
                output.write(json.dumps({'id':line['id'], 'data':line[tq_name], 'task':line['task'], 'label':p_lab}, ensure_ascii=False)+'\n')
        else:    # ner任务
            for line in task_data_list:
                prompt = ['犯罪嫌疑人情况','被害人','被害人类型','犯罪嫌疑人交通工具','犯罪嫌疑人交通工具情况','被害人交通工具情况','犯罪嫌疑人责任认定','被害人责任认定','事故发生地','被害人交通工具']
                p_lab = {}
                ie.set_schema(prompt)
                result = ie(line[tq_name].strip())
                for entity_type in prompt:
                    p_lab[entity_type] = []
                    if entity_type not in result[0]:
                        p_lab[entity_type].append([])
                        continue
                    for samp in result[0][entity_type]:
                        p_lab[entity_type].append([samp['start'], samp['end']])
                output.write(json.dumps({'id':line['id'], 'data':line[tq_name], 'task':line['task'], 'label':p_lab}, ensure_ascii=False)+'\n')
    output.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=1, help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--input_file", default="./input/test", help="Whether to run training.") 
    parser.add_argument("--output_file", default="./result/output", help="Whether to run training.")  
    args = parser.parse_args()

    predict()

