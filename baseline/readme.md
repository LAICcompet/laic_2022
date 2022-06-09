# LAIC2022



## 模型

1. + Roformer+GlobalPointer

## 数据集

1. 训练时将整理好的json数据集放在`./input_data`目录下。



## 训练

1. 训练参数可在`./config/config.yml`中调整，你必须在`main.py`的参数中选择do_train还是do_predict。

2. 训练：

   ```python
   python main.py --do_train
   
   ```

3、训练成果：

训练完成后，最优的模型保存在`./best_model`中，可以在config.yml中进行路径调整。

## 测试

1. 封闭评测时，需在`main.sh`中关闭`--do_train`，开启 `--do_predict`，预测数据`--test_file`请从`../input/input.json`读取，需要提交的文件`--output_file`保存为`../output/output.json`。