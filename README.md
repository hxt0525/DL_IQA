# 使用说明
  该代码用于基于深度学习的IQA任务的训练。<br>
  关键代码在IQAtools文件夹中。<br>
  运行采用配置文件的方式，所有进行过实验的配置文件在configs文件夹中，可根据该文件夹中的配置文件进行相应的配置修改。<br>
  label_split文件夹中包含进行训练验证的图片的标签，算法模型已在CSIQ、LIVE、TID2013、KonIQ、LIVEC、SPAQ上进行了运行验证，该文件夹中包含了具体的训练集和验证集的划分。<br>
# 实验结果
  按照8：2进行训练集和验证集划分，结果如下，具体训练细节请查看相关配置文件
|SROCC|SPAQ|KonIQ|LIVEC|
| :--- |:----:|---:|---:|
|EfficientB2MulFeatureWeightNet|	0.919|	0.915|	0.871|
|EfficientB2MulFeatureWeightNetMall|	0.919|	0.925|	0.892|

# 训练
单卡 python train.py --cfg_path 配置文件路径<br>
多卡 python -m torch.distributed.launch --nproc_per_node=使用的GPU数目 --master_port 自定义设置的主进程ID --cfg_path 配置文件路径
