import json
import sys

def load_json(filename):
    """加载数据
    单条格式：[id, text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    dataset = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            d = []
            d.append(l["id"])
            d.append(l["context"])
            for entity in l["entities"]:
                label = entity["label"]
                for span in entity["span"]:
                    if type(span) == str:
                        span_list = span.split(";")
                        d.append([int(span_list[0]),int(span_list[1]),label])
                    elif type(span) == list:
                        d.append([span[0], span[1], label])
            dataset.append(d)
    return dataset

def evaluate(true_path,output_path, log):
    try:
        data = load_json(true_path)
        predict_data = load_json(output_path)
        """评测函数
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in data:
            # 找到predict_data 中和d同样的数据json，记作R, 对比R和T
            for one in predict_data:
                if one[0] == d[0]:
                    R = set([tuple(n) for n in one[2:]])
            T = set([tuple(i) for i in d[2:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        print({'f1':f1, 'precision':precision , 'recall':recall}, file=open(log, "w", encoding="utf-8"))
        # return {'f1':f1, 'precision':precision , 'recall':recall}
    except:
        print({'f1': -1, 'precision': -1, 'recall': -1}, file=open(log, "w", encoding="utf-8"))

        # return {'f1': -1, 'precision': -1, 'recall': -1}

if __name__ == '__main__':

    # true_path = '..\\data\\test.json'
    # output_path = 'output'
    evaluate(sys.argv[1], sys.argv[2],sys.argv[3])

