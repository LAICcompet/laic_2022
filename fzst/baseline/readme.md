# LAIC2022



## 模型

1. + Roformer+GlobalPointer

## 数据集

1. 训练时将整理好的json数据集放在`./input_data`目录下。



## 训练

1. 训练参数可在`./config/config.yml`中调整。

2. 训练：

   ```python
   python train.py
   ```

3、训练成果：

训练完成后，最优的模型保存在`./best_model`中，可以在config.yml中进行路径调整。

## 测试

1. 封闭评测时执行以下代码

   ```
   python main.py \
   --input_file "./input_data/test.json" \
   --output_file "./output/result.json" \
   ```