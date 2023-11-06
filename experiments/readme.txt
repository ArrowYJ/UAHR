此次更新了train.py, main_mutil.py,Dataset_cluster.py, train_GPU_4_0.sh文件

需要下载 resnet50的预训练参数：https://download.pytorch.org/models/resnet50-19c8e357.pth

将下载参数的地址填写到 train_GPU_4_0.sh 文件的 --cluster_checkpoint 参数后即可

（代码中，模型加载参数的位置在train.py 的 353 行）


这一次测试运行：

train_GPU_4_0.sh

目前先检测一下是否能成功运行到完成预训练模型加载（约3~4个epoch），检验完毕后，再去除各种日志写入
