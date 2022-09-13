# LAIC2022

## 模型

1. + uie-base

## 数据集

1. 通过运行 ``./doccano.py``程序将 ``./input``下的10个任务数据集生成 `./data`目录下多任务训练数据集。

## 训练

1. 训练参数可在 `./finetune.py`中调整，在 `./start.sh`中选择单卡、多卡启动。
2. 训练：

   ```python
   python finetune.py 或 bash start.sh
   ```

3、训练成果：

训练完成后，最优的模型保存在 `./checkpoint/best_model`中，可以在 ``finetune.py``中进行路径参数调整。
