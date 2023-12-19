此次更新了 train_GPU_4_0.sh, train_GPU_4_1.sh, train_GPU_4_2.sh, train_GPU_4_3.sh, train_GPU_4_4.sh, 
\experiments\data\Dataset_cluster.py, experiments\main_stage_0.py, lifelong_methods\methods\base_method.py,
lifelong_methods\models\resnet.py, lifelong_methods\methods\New.py  文件

这一次在这5个脚本里，仍需将resnet50的预训练参数地址添加到 --cluster_checkpoint 中。

这一次调参运行：

train_GPU_4_0.sh
train_GPU_4_1.sh
train_GPU_4_2.sh
train_GPU_4_3.sh
train_GPU_4_4.sh

目前已经可以保证首阶段不掉点，就看最后的性能表现了   2023/12/20

------------------------------------------------------------------------------------------------------------------
此次更新了 train_GPU_4_0.sh, train_GPU_4_1.sh, train_GPU_4_2.sh, train_GPU_4_3.sh, train_GPU_4_4.sh, 
\lifelong_methods\methods\New.py, experiments\train.py 文件

这一次在这5个脚本里，仍需将resnet50的预训练参数地址添加到 --cluster_checkpoint 中。

这一次调参运行：

train_GPU_4_0.sh
train_GPU_4_1.sh
train_GPU_4_2.sh
train_GPU_4_3.sh
train_GPU_4_4.sh

麻烦师兄了！万分感谢！

----------------------------------------------------------------------------------------------------------------------
需要下载 resnet50的预训练参数：https://download.pytorch.org/models/resnet50-19c8e357.pth

将下载参数的地址填写到 train_GPU_4_0.sh 文件的 --cluster_checkpoint 参数后即可

（代码中，模型加载参数的位置在train.py 的 353 行）


这一次测试运行：

train_GPU_4_0.sh

目前先检测一下是否能成功运行到完成预训练模型加载（约3~4个epoch），检验完毕后，再去除各种日志写入
