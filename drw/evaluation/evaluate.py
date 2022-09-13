#coding:utf-8
import json,os
import sys


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
        X, Y, Z = 1e-10, 1e-10, 1e-10
        if len(real_data) != len(predict_data):
            print('预测样本部分缺失，请检查！')
        else:
            for real in real_data:
                if real_data[real]['task'] == 'NER':
                    for entity_name in real_data[real]['label']:
                        pre_lab_lis = predict_data[real]['label'][entity_name]
                        X += len(set(real_data[real]['label'][entity_name]) & set(pre_lab_lis))
                        Y += len(pre_lab_lis)
                        Z += len(real_data[real]['label'][entity_name])
                else:
                    pre_lab_lis = predict_data[real]['label']
                    X += len(set(real_data[real]['label']) & set(pre_lab_lis))
                    Y += len(pre_lab_lis)
                    Z += len(real_data[real]['label'])
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return {'f1': f1, 'precision': precision, 'recall': recall}
    except:
        return {'f1': -1, 'precision': -1, 'recall': -1}



if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])

