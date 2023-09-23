import argparse
from cProfile import label
import importlib
import logging
import os
import sys
import random

import torch
import torchvision
from ml_logger import logbook as ml_logbook
import time

import numpy as np
import math
from math import ceil
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from iirc.datasets_loader import get_lifelong_datasets
from iirc.utils.T_SNE import draw_tsne
from lifelong_methods.utils import transform_labels_names_to_vector
import lifelong_methods.utils
import lifelong_methods
import experiments.utils
import pdb
from tqdm import tqdm
from iirc.utils.Gradcam import draw_gradcam
import torch.utils.data as data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# get the model of the first task                  
# get the feature of each class in session 1 
# calculate the MSV value for each class
# observe the difference betwwen class that has no subclass and the class that has different numbers subclasses

def my_std(matrix_X, avg_vec):
    dist_each_row = matrix_X - avg_vec
    std_x = np.sqrt(np.sum(np.sum(dist_each_row * dist_each_row, axis=1)))
    return std_x

def MSV_Single_Class(feature_list):
    # one row represents one sample
    feature_all = np.array(feature_list)
    feature_avg = np.average(feature_all, axis=0)
    feature_dis = feature_all - feature_avg
    feature_dis = np.sum(np.maximum(feature_dis, -feature_dis), axis=1)
    feature_idx = np.argsort(feature_dis)
    start_idx = math.ceil(float(feature_idx.size) * 0.25)  # remove the first 1 / 4 data
    end_idx = math.floor(float(feature_idx.size) * 0.75)  # remove the last 1 / 4 data
    feature_correct = feature_all[feature_idx[start_idx:end_idx]]
    return my_std(feature_correct, feature_avg)

def MSV_Mutil_Class(feature_all):
    feature_all = np.array(feature_all)
    class_num = feature_all.shape[0]
    ans_msv = []
    for i in range(class_num):
        ans_msv.append(MSV_Single_Class(feature_all[i, :, :]))
    return ans_msv

def get_gradcam_transforms():
    essential_transforms_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
    ])
    return essential_transforms_fn, augmentation_transforms_fn

def get_transforms(dataset_name):
    essential_transforms_fn = None
    augmentation_transforms_fn = None
    if "cifar100" in dataset_name:
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return essential_transforms_fn, augmentation_transforms_fn

def get_the_dataset(config, task_id):
    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root='./../data',
                              tasks_configuration_id=config["tasks_configuration_id"],
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=config["joint"])
    if config["complete_info"]:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()
    
    if config["incremental_joint"]:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.load_tasks_up_to(task_id)
    else:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.choose_task(task_id)

    task_train_data = lifelong_datasets['train']
    task_valid_data = lifelong_datasets['intask_valid']
    
    #pdb.set_trace()

    train_loader = data.DataLoader(
        task_train_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    valid_loader = data.DataLoader(
        task_valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    return train_loader, valid_loader, tasks, class_names_to_idx

def get_the_model(checkpoint_path, task_id):
    checkpoint = torch.load(f'{checkpoint_path}{task_id}_model')
    config = checkpoint['config']
    metadata = checkpoint['metadata']

    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)
    model.load_method_state_dict(checkpoint["method_state_dict"])

    return model, config

def convert_label_vector_to_idx(label_vector):
    label_index = np.argmax(label_vector, axis=1)
    return label_index

def start_my_test(checkpoint_path, task_id):
    model, config = get_the_model(checkpoint_path=checkpoint_path, task_id=task_id)
    train_loader, valid_loader, tasks, class_names_to_idx = get_the_dataset(config, task_id)
    num_seen_classes = len(model.seen_classes)
    feature_bank = np.array([0.0] * 64)   #------------------------------------64: the length of latent feature
    precision_bank = np.array([0] * num_seen_classes)
    label_bank = np.array([0] * num_seen_classes)
    model.to(config["device"])
    model.net.eval()

    with torch.no_grad():
        for minibatch in tqdm(valid_loader):
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )
            images = minibatch[0].to(config["device"], non_blocking=True)
            labels = labels.to(config["device"], non_blocking=True)
            output, latent_feat = model.forward_net(images)
            pdb.set_trace()
           # precision = output.ge(0.0)
            
            feature_bank = np.vstack((feature_bank, latent_feat.cpu().numpy()))
            precision_bank = np.vstack((precision_bank, output.cpu().numpy()))
            label_bank = np.vstack((label_bank, labels.cpu().numpy()))
    
    feature_bank = feature_bank[1:]
    label_bank = label_bank[1:]
    precision_bank = precision_bank[1:]

    label_index = convert_label_vector_to_idx(label_bank)
    
    precision_index = convert_label_vector_to_idx(precision_bank)
    all_count = len(label_index)
    correct_count = 0

    for i in range(len(label_index)):
        if precision_index[i] == label_index[i]:
            correct_count = correct_count + 1
    print(f'{task_id} acc: {float(correct_count) / all_count}')
    
    tag = []
    for i in range(task_id + 1):
        tag.extend(tasks[i])
    draw_tsne(X=feature_bank, labels=label_index, show_number=50, tag=tag, output_dir='./../my_result/lucir/', title=f'{task_id}_tsne')
    msv_value = []
    for i in range(num_seen_classes):
        temp_x = np.where(label_index == i)[0]
        if len(temp_x) == 0:
            msv_value.append(0)
            continue
        single_feature = feature_bank[temp_x, :]
        msv_value.append(MSV_Single_Class(single_feature))
    msv_value = np.array(msv_value)
    np.savetxt(f'./../my_result/lucir/{task_id}_msv.txt', msv_value, fmt="%f", delimiter=" ")
   # pdb.set_trace()



if __name__ == '__main__':
    model_name = 'iirc_cifar100_lucir_momentum_lr1.0_wd1e-05_tci0_nmpc20_lr_on_plat_use_best_nl32_ept100_s100_54608/'
    checkpoint_path = './../results/' + model_name
    for i in range(22):  # 22 is the number of the tasks
        start_my_test(checkpoint_path=checkpoint_path, task_id=i)




    