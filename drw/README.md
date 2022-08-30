# LAIC2022——小样本多任务学习

该项目为 **LAIC2022—小样本多任务学习** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://data.court.gov.cn/pages/laic2021.html)**（待修改）**。该数据只可用于该比赛，未经允许禁止在其他领域中使用。

## 选手交流群(待修改)

QQ群：237633234

## 数据说明

针对本次任务，我们会提供包含案件情节描述的陈述文本，选手需要识别出文本中的关键信息，并按照规定格式返回结果。
数据需要doccano.py文件经过转换才能用于训练。

##### 1. 案件要素

案件要素中包含8个二分类子任务，子任务数据格式为json格式，字典包含字段为：

- ``data``：文本内容，文书中事实描述部分。
- ``label``：是否是当前案件要素，是/否。

##### 2. 刑档

刑档任务数据格式为json格式，字典包含字段：

- ``data``：文本内容，文书中事实描述部分。
- ``label``：文书对应刑档等级，一档/二档/三档。

##### 3. 命名实体识别

命名实体识别任务数据格式为json格式，字典包含字段：

- ``id``：案例中文本的唯一标识符。
- ``text``：文本内容，文书中事实描述部分。
- ``entities``：句子所包含的实体列表。
- ``label``：实体标签名称。
- ``start_offset``：实体开始位置下标。
- ``end_offset``：实体结束位置下标。
- ``relations``：空，无用。

其中``命名实体识别``任务中的``label``的十种实体类型分别为：

|label|含义|
|---|---|
|11017|犯罪嫌疑人情况|
|11018|被害人|
|11019|被害人类型|
|11020|犯罪嫌疑人交通工具|
|11021|犯罪嫌疑人交通工具情况|
|11022|被害人交通工具情况|
|11023|犯罪嫌疑人责任认定|
|11024|被害人责任认定|
|11025|事故发生地|
|11027|被害人交通工具|

**注意：**

此次赛题不允许通过添加额外有监督样本数量提升模型的预测效果

## 提交文件格式及组织形式

在模型评测阶段，你需要将所有的代码压缩为一个必须为`zip`的文件进行提交，该`zip`文件将通过**内部**`main.py`文件将包含的程序包含在内作为运行的。在评测阶段会使用如下命令来运行你的程序：

```python
python main.py \
--test_file "default" \
--output_file "default" \
```

**注意：**

（1）main.py文件必须包含2个参数,接收测试输入文件和结果保存文件名称。

（2）上传的压缩包命名方式请参考大赛官方说明。

（3）请使用python3版本。



## 预测结果输出格式

你需要从`测试输入文件`中读取数据进行预测，此数据格式和下发数据的格式完全一致，通过运行模型，将预测的结果文件输出到`结果保存文件`中，此输出的数据格式可以参考`evaluation/sample.txt`。

**请注意：**

（1）具体的提交评测说明见baseline目录下的`README.md`文件。

（2）我们将可提交模型大小限制在2G以内，请进一步地精简模型。

## 评测脚本

在 `evaluation` 文件夹中提供了评分的代码和提交文件样例，以供参考

## 模型模型

我们提供了简单的方法供选手参考，模型放在`baseline`目录。

## 动态模型运行环境

```
参考baseline文件夹下的requirements.txt
```

## 问题反馈（待修改）

如有问题，请在QQ群：521382653中反馈