#coding:utf-8
import json,os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

def load_predictData(path):
    predict_dir = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_js = json.loads(line)
            labels_dir = {}
            if line_js['task'] == 'NER':
                for entity,start_end_lis in line_js['label'].items():
                    start_end_ = [tuple(loc) for loc in start_end_lis]
                    labels_dir[entity] = start_end_
                predict_dir[line_js['id']] = {'task':line_js['task'], 'data':line_js['data'], 'label':labels_dir}
            else:
                lab = line_js['label'][0]
                predict_dir[line_js['id']] = {'task':line_js['task'], 'data':line_js['data'], 'label':[tuple(lab)]}
    return predict_dir
            
def load_trueData(path):
    data_dir = {}
    for file_name in os.listdir(path):
        if file_name == '刑档':
            with open(os.path.join(path, file_name), encoding='utf8') as f:
                for line in f:
                    line_js = json.loads(line)
                    data_dir[line_js['id']] = {'task':line_js['task'], 'data':line_js['data'], 'label':[tuple(line_js['label'])]}

        elif file_name == '案件要素':
            for case_plot_name in os.listdir(os.path.join(path, file_name)):
                with open(os.path.join(path, os.path.join(file_name, case_plot_name)), encoding='utf8') as f:
                    for line in f:
                        line_js = json.loads(line)
                        data_dir[line_js['id']] = {'task':line_js['task'], 'data':line_js['data'], 'label':[tuple(line_js['label'])]}

        elif file_name == 'ner':
            with open(os.path.join(path, file_name), encoding='utf8') as f:
                for line in f:
                    line_js = json.loads(line)
                    # 统计实体
                    entities = {}
                    for entity_dir in line_js["entities"]:
                        start_end = [entity_dir['start_offset'], entity_dir['end_offset']]
                        entities.setdefault(entity_dir['label'], []).append(tuple(start_end))
                    data_dir[line_js['id']] = {'task':line_js['task'], 'data':line_js['text'], 'label':entities}
    return data_dir


def evaluate(true_path, output_path): 
    try:
        real_data = load_trueData(true_path)
        predict_data = load_predictData(output_path)
        """评测函数"""
        if len(real_data) != len(predict_data):
            return {'precision': -1, 'recall': -1, 'f1': -1}
        X, Y, Z = 1e-10, 1e-10, 1e-10
        cls_data_dir = {'xingdang':{'true_result':[], 'predict_result':[]},
                        'case_elements':{'true_result':[], 'predict_result':[]}}
        for real in real_data:
            if real>=0 and real<=99:   # NER
                for entity_name in real_data[real]['label']:
                    pre_lab_lis = predict_data[real]['label'][entity_name]
                    X += len(set(real_data[real]['label'][entity_name]) & set(pre_lab_lis))
                    Z += len(real_data[real]['label'][entity_name])
                    if not pre_lab_lis[0]:
                        continue
                    Y += len(pre_lab_lis)
            elif real>=100 and real<=159:  # 刑档
                cls_data_dir['xingdang']['true_result'].append(real_data[real]['label'][0][0])
                if predict_data[real]['label'][0]:
                    cls_data_dir['xingdang']['predict_result'].append(predict_data[real]['label'][0][0])
                else:
                    temp_lis = [e for e in ['一档','二档','三档'] if e != real_data[real]['label'][0][0]]
                    cls_data_dir['xingdang']['predict_result'].append(temp_lis[0])
            elif real>=160 and real<=372:  # 案件要素
                cls_data_dir['case_elements']['true_result'].append(real_data[real]['label'][0][0])
                if predict_data[real]['label'][0]:
                    cls_data_dir['case_elements']['predict_result'].append(predict_data[real]['label'][0][0])
                else:
                    temp_lis = [e for e in ['正','负'] if e != real_data[real]['label'][0][0]]
                    cls_data_dir['case_elements']['predict_result'].append(temp_lis[0])
        f1_NER, precision_NER, recall_NER = 2 * X / (Y + Z), X / Y, X / Z

        f1_CLS_case_elements = f1_score(cls_data_dir['case_elements']['true_result'], cls_data_dir['case_elements']['predict_result'], pos_label='正')
        precision_CLS_case_elements = precision_score(cls_data_dir['case_elements']['true_result'], cls_data_dir['case_elements']['predict_result'], pos_label='正')
        recall_CLS_case_elements = recall_score(cls_data_dir['case_elements']['true_result'], cls_data_dir['case_elements']['predict_result'], pos_label='正')

        f1_CLS_xingdang = f1_score(cls_data_dir['xingdang']['true_result'], cls_data_dir['xingdang']['predict_result'], average='macro')
        precision_CLS_xingdang = precision_score(cls_data_dir['xingdang']['true_result'], cls_data_dir['xingdang']['predict_result'], average='macro')
        recall_CLS_xingdang = recall_score(cls_data_dir['xingdang']['true_result'], cls_data_dir['xingdang']['predict_result'], average='macro')

        return {'precision': (precision_NER + precision_CLS_case_elements + precision_CLS_xingdang)/3, 
                'recall': (recall_NER + recall_CLS_case_elements + recall_CLS_xingdang)/3,
                'f1': (f1_NER + f1_CLS_case_elements + f1_CLS_xingdang)/3}
    except:
        return {'precision': -1, 'recall': -1, 'f1': -1}



if __name__ == '__main__':
    result = evaluate(sys.argv[1], sys.argv[2])
    print(result)


