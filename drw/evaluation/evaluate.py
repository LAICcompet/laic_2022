#coding:utf-8
import json
import sys
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def load_predictData(filename):
    """加载数据
    单条格式：{'data':text, 'labels':[{start,end}, {start,end}...{start,end}]}，
              意味着labels中可能存在一个或多个{start,end}，start和end为'命名实体识别'的
              实体在文本中的位置下标或分类的类别名称（表示为：[{'正'}]/[{'一档'}]）。
    """
    predict_lis = []
    with open(filename, encoding='utf-8') as f:
        for lin in f:
            lin_js = json.loads(lin)
            predict_lis.append([tuple(la_set) for la_set in eval(lin_js['labels'])])
    return predict_lis
            
def load_trueData(filename):
    true_lis = []
    with open(filename, encoding='utf-8') as f:
        for lin in f:
            lin_js= json.loads(lin)
            if lin_js['result_list'][0]['text'] in ['正','负','一档','二档','三档']:
                true_lis.append([tuple({lin_js['result_list'][0]['text']})])
            else:
                true_lis.append([tuple({start_end['start'], start_end['end']}) for start_end in lin_js['result_list']])
    return true_lis

def evaluate(true_path,output_path, log):
    try:
        real_label = load_trueData(true_path)
        predict_label = load_predictData(output_path)
        """评测函数"""
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for real,pre in zip(real_label, predict_label):
            X += len(set(real) & set(pre))
            Y += len(pre)
            Z += len(real)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        print({'f1':f1, 'precision':precision , 'recall':recall}, file=open(log, "w", encoding="utf-8"))
    except:
        print({'f1': -1, 'precision': -1, 'recall': -1}, file=open(log, "w", encoding="utf-8"))


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2],sys.argv[3])
