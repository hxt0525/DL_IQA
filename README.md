# 使用说明
  该代码用于基于深度学习的IQA任务的训练
  关键代码在IQAtools文件夹中
  运行采用配置文件的方式，所有进行过实验的配置文件在configs文件夹中，可根据该文件夹中的配置文件进行相应的配置修改
  label_split文件夹中包含进行训练验证的图片的标签，算法模型已在CSIQ、LIVE、TID2013、KonIQ、LIVEC、SPAQ上进行了运行验证，该文件夹中包含了具体的训练集和验证集的划分
# 实验结果
  按照8：2进行训练集和验证集划分，结果如下，具体训练细节请查看相关配置文件
|SROCC|SPAQ|KonIQ|LIVEC|
| :--- |:----:|---:|---:|
|EfficientB2MulFeatureWeightNet|	0.919|	0.915|	0.871|
|EfficientB2MulFeatureWeightNetMall|	0.919|	0.925|	0.892|

# Training on IQA databases
python train.py --cfg_path your_cfg.yaml
