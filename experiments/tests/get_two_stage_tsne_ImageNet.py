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
#from iirc.utils.Gradcam import draw_gradcam
import torch.utils.data as data
from lifelong_methods.utils import SubsetSampler

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
        get_lifelong_datasets(config['dataset'], dataset_root='./../data/imagenet/',
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
    
    select_class = []
    new_tasks = [['kitchen_appliances', 'curtain-screen', 'fungus', 'box', 'garment', 'gymnastic_apparatus', 'bus', 'green_groceries', 'table', 'measuring_instrument'], ['n07753275', 'n03982430', 'n02777292', 'n03769881', 'n03544143']]
    if task_id == 0:
        select_class = new_tasks[0]
    else:
        select_class = new_tasks[1]
    data_indices = [0]
    for class_ in select_class:
        class_data_indices = task_valid_data.get_image_indices_by_cla(class_)
        data_indices.extend(list(class_data_indices))
    data_indices = np.array(data_indices[1:])
    sampler = SubsetSampler(data_indices)

    train_loader = data.DataLoader(
        task_train_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], sampler=sampler
    )
    valid_loader = data.DataLoader(
        task_valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], sampler=sampler
    )
 #   pdb.set_trace()
    return train_loader, valid_loader, new_tasks, class_names_to_idx

def get_the_model(checkpoint_path, task_id):
    checkpoint = torch.load(f'{checkpoint_path}{task_id}_model')
    config = checkpoint['config']
    metadata = checkpoint['metadata']

    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)
    model.load_method_state_dict(checkpoint["method_state_dict"])

    return model, config

def transform_labels_names_to_idx(label_names, class_names_to_idx):
   label_idxs = []
   for names in label_names:
       label_idxs.append(class_names_to_idx[names[0]])
   return label_idxs


def convert_label_vector_to_idx(label_vector):
    label_index = np.argmax(label_vector, axis=1)
    return label_index

def start_my_test(checkpoint_path, task_id):
    #get all test models
    test_model = []
    model_task_id = []
    model_num = len(checkpoint_path)
    task_num = len(task_id)
    for i in range(model_num):
        for j in range(task_num):
            model1, config = get_the_model(checkpoint_path=checkpoint_path[i], task_id=task_id[j])
            test_model.append(model1)
            model_task_id.append(task_id[j])
    feature_bank = []
    label_bank = []
    for i in range(len(test_model)):
        test_model[i].to(config["device"])
        test_model[i].net.eval()
        feature_bank.append(np.array([0.0] * 2048))   #------------------------------------64: the length of latent feature
        label_bank.append([0])

    for data_idx in task_id:
        _, valid_loader, tasks, class_names_to_idx = get_the_dataset(config, data_idx)
        with torch.no_grad():
            for minibatch in tqdm(valid_loader):
                labels_names = list(zip(minibatch[1], minibatch[2]))
                labels = transform_labels_names_to_idx(
                    labels_names, class_names_to_idx
                )
                images = minibatch[0].to(config["device"], non_blocking=True)
                for i in range(len(test_model)):
                    if model_task_id[i] < data_idx:
                        continue
                    _, latent_feat = test_model[i].forward_net(images)
                    feature_bank[i] = np.vstack((feature_bank[i], latent_feat.cpu().numpy()))
                    label_bank[i].extend(labels)
    
    for i in range(len(feature_bank)):
        feature_bank[i] = feature_bank[i][1:]
        label_bank[i] = label_bank[i][1:]
    
    # get the feature of the test model and the valid data
    tag = []
    tag.append(tasks[0][:])
    tag.append(tasks[0][:])
    tag[1].extend(tasks[1][:])
#    tag.extend(tasks[0])
#    tag.extend(tasks[1])
    #for task_idx in task_id:
    #    tag.extend(tasks[task_idx])
    #pdb.set_trace()
    min_output_dir = ['model_0_task_0/', 'model_0_task_1/', 'model_1_task_0/', 'model_1_task_1/']
    for randx in tqdm(range(100)):
        for i in range(model_num):
            for j in range(task_num):
                cur_label = np.array(label_bank[i * task_num + j])
                cur_fea = feature_bank[i * task_num + j]
                cur_tag = tag[j]

                draw_tsne(X=cur_fea, labels=cur_label, show_number=25, tag= cur_tag, output_dir='./../my_result/' + min_output_dir[i * task_num + j], title=f'rank_{randx}_model_{i}_task_{j}_tsne')
    



if __name__ == '__main__':
    temp_checkpoint_path_dir = './../server_result/'
    checkpoint_path = []
    checkpoint_path.append(temp_checkpoint_path_dir + 'IIRC_NORM_3090_0/')   
    checkpoint_path.append(temp_checkpoint_path_dir + 'IIRC_REFINE_3090_0/')
    test_task_id = [0, 7]

    start_my_test(checkpoint_path=checkpoint_path, task_id=test_task_id)




    