
#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/No_and_have/figure/ground_truth' --tasks_configuration_id 0

CUDA_VISIBLE_DEVICES=0 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/confusion_matrix/ground_truth' --tasks_configuration_id 0

CUDA_VISIBLE_DEVICES=0 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/confusion_matrix/cluster_55_inter_10_mvalue_0.75_nvalue_0.15' --models_dir './my_result/model/cluster_55_inter_10_mvalue_0.75_nvalue_0.15' --tasks_configuration_id 0

#-------------------------------------------------------

#CUDA_VISIBLE_DEVICES=0 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/confusion_matrix/cluster_55_inter_7_mvalue_0.50_nvalue_0.10_loss_ranking' --models_dir './my_result/model/cluster_55_inter_7_mvalue_0.50_nvalue_0.10_loss_ranking' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=0 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/confusion_matrix/cluster_55_inter_7_mvalue_0.50_nvalue_0.10_loss_divide' --models_dir './my_result/model/cluster_55_inter_7_mvalue_0.50_nvalue_0.10_loss_divide' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=0 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/confusion_matrix/cluster_55_inter_7_mvalue_0.50_nvalue_0.10' --models_dir './my_result/model/cluster_55_inter_7_mvalue_0.50_nvalue_0.10' --tasks_configuration_id 0

#-----------------------------------------------

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/No_and_have/figure/iirc_cifar100_icarl_refine_2.5_0_0.1' --models_dir './my_result/No_and_have/iirc_cifar100_icarl_refine_2.5_0_0.1' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/No_and_have/figure/iirc_cifar100_icarl_refine_2.5_10_0' --models_dir './my_result/No_and_have/iirc_cifar100_icarl_refine_2.5_10_0' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/No_and_have/figure/iirc_cifar100_icarl_refine_2.5_10_0.1' --models_dir './my_result/No_and_have/iirc_cifar100_icarl_refine_2.5_10_0.1' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_cifar100 --dataset_path './data' --results_path './my_result/No_and_have/figure/fine_tune' --models_dir './my_result/No_and_have/fine_tune' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_imagenet_lite --dataset_path './data/imagenet/' --results_path './confusion/ground/' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_imagenet_lite --dataset_path './data/imagenet/' --results_path './confusion/norm/' --models_dir './server_result/IIRC_NORM_3090_0/' --tasks_configuration_id 0

#CUDA_VISIBLE_DEVICES=1 python create_confusion_matrix.py --dataset iirc_imagenet_lite --dataset_path './data/imagenet/' --results_path './confusion/refine/' --models_dir './server_result/IIRC_REFINE_3090_0/' --tasks_configuration_id 0
