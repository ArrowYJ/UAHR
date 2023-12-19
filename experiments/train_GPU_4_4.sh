 CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch  --master_port 12589 --nproc_per_node=4 main_mutil.py \
 --dataset_path "./data/imagenet"  --rbf_dis 0.001 --l_divide 50.0  --l_margin 0.15 --temperature 2.0  --my_logging './refine' \
 --use_cluster_dataset  --cluster_checkpoint "None"  --cluster_model "resnet50" --K_Cluster 50 --inter_cluster 23 --m_value 0.50 \
 --n_value 0.10 --end_cluster_task 1 --class_value 0.08 --test_incre 0 --dataset iirc_imagenet_full --epochs_per_task 100 \
 --single_batch_size 128 --seed 100 --logging_path_root "./results" --n_layers 50 --tasks_configuration_id 0 --method New \
 --optimizer momentum --lr 0.4 --lr_gamma 0.1 --reduce_lr_on_plateau --weight_decay 1e-5 --n_memories_per_class 20 \
 --checkpoint_interval 10 # --use_best_model