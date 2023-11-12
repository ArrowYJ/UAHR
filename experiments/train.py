import os
import time
import torch
import torch.utils.data as data
import torch.distributed as dist
from copy import deepcopy

import torchvision.models as models
import torch.nn as nn

from iirc.utils.utils import print_msg
import lifelong_methods.utils
from lifelong_methods.buffer.buffer import TaskDataMergedWithBuffer
from lifelong_methods.utils import transform_labels_names_to_vector
from experiments.data.Dataset_cluster import ClusteringDataset, SameClassSampler, SameClassBatcherSampler, DDPClusteringDataset, DDPBaseBatchSampler
from experiments import utils, metrics
from iirc.utils.Mysampler import EachClassSampler_Buffer, EachClassBatchSampler_Buffer, SingleClassSampler_Buffer, Support_ClassSampler_Buffer
import pdb
from itertools import cycle
from tqdm import tqdm

def Support_epoch_train(model, dataloader, config, metadata, gpu=None, rank=0, epoch_cur = 0, cur_task_id=0):
    train_loss = 0
    train_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.train()
    minibatch_i = 0
    for i, data in enumerate(zip(tqdm(dataloader[0]), cycle(dataloader[1]))):
        # query train data 
        
        query_data = data[0]
        query_labels_names = list(zip(query_data[1], query_data[2]))
        query_labels = transform_labels_names_to_vector(
            query_labels_names, num_seen_classes, class_names_to_idx
        )
        if gpu is None:
            query_images = query_data[0].to(config["device"], non_blocking=True)
            query_labels = query_labels.to(config["device"], non_blocking=True)
        else:
            query_images = query_data[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            query_labels = query_labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

        if len(query_data) > 3:
            if gpu is None:
                in_buffer = query_data[3].to(config["device"], non_blocking=True)
            else:
                in_buffer = query_data[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True, requires_grad=True)
        else:
            in_buffer = None
        
        # support train data
        support_data = data[1]
        support_labels_names = list(zip(support_data[1], support_data[2]))
        support_labels = transform_labels_names_to_vector(
            support_labels_names, num_seen_classes, class_names_to_idx
        )
        if gpu is None:
            support_images = support_data[0].to(config["device"], non_blocking=True)
            support_labels = support_labels.to(config["device"], non_blocking=True)
        else:
            support_images = support_data[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            support_labels = support_labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)
        
    #    pdb.set_trace()
        # start train

        predictions, loss = model.observe(query_images, query_labels, support_images, support_labels, in_buffer, train=True, epoch_cur=epoch_cur)

        labels = query_labels.bool()
        train_loss += loss * query_images.shape[0]

        train_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * query_images.shape[0]
        train_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * query_images.shape[0]
        train_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * query_images.shape[0]
        train_metrics['recall'] += metrics.recall(predictions, labels) * query_images.shape[0]
        data_len += query_images.shape[0]

        if minibatch_i == 0:
            print_msg(
                f"rank {rank}, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB, "
                f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB\n")
        minibatch_i += 1
        
        
    model._save_task_prototype(dataloader[1])
    
    train_loss /= data_len
    train_metrics['jaccard_sim'] /= data_len
    train_metrics['modified_jaccard'] /= data_len
    train_metrics['strict_acc'] /= data_len
    train_metrics['recall'] /= data_len
    
    
    return train_loss, train_metrics


def epoch_train(model, dataloader, config, metadata, gpu=None, rank=0, epoch_cur = 0):
    train_loss = 0
    train_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.train()
    minibatch_i = 0
    for minibatch in dataloader:  
        labels_names = list(zip(minibatch[1], minibatch[2]))
        labels = transform_labels_names_to_vector(
            labels_names, num_seen_classes, class_names_to_idx
        )

        if gpu is None:
            images = minibatch[0].to(config["device"], non_blocking=True)
            labels = labels.to(config["device"], non_blocking=True)
        else:
            images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

        if len(minibatch) > 3:
            if gpu is None:
                in_buffer = minibatch[3].to(config["device"], non_blocking=True)
            else:
                in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)  #, requires_grad=True
        else:
            in_buffer = None
        if config["method"] == 'lucir_duq' or config['method'] == "duq_my":
            predictions, loss = model.observe(images, labels, in_buffer, train=True, epoch_cur=epoch_cur)
        else:
            predictions, loss = model.observe(images, labels, in_buffer, train=True)

        labels = labels.bool()
        train_loss += loss * images.shape[0]
        train_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
        train_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
        data_len += images.shape[0]
        
        if minibatch_i == 0:
            print_msg(
                f"rank {rank}, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB, "
                f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB\n")
        minibatch_i += 1

    train_loss /= data_len
    train_metrics['jaccard_sim'] /= data_len
    train_metrics['modified_jaccard'] /= data_len
    train_metrics['strict_acc'] /= data_len
    train_metrics['recall'] /= data_len
    return train_loss, train_metrics


def Cluster_epoch_train(model, dataloader, config, metadata, gpu=None):
    model.net.train()
    loss_avg = 0.0
    data_len = 0
    for minibatch in dataloader:  
        labels_names = list(minibatch[2])
        labels = torch.zeros((len(labels_names), config["K_Cluster"]))
        for i in range(len(labels)):
            labels[i, labels_names[i]] = 1

        if gpu is None:
            images = minibatch[0].to(config["device"], non_blocking=True)
            labels = labels.to(config["device"], non_blocking=True)
        else:
            images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            
        loss = model.cluster_train(images, labels, train=True)
        loss_avg = loss_avg + loss
        data_len = data_len + 1
    #print_msg(f"Each class has {config['K_Cluster']} class centers. the loss is {loss_avg / data_len}")
    

def evaluate(model, dataloader, config, metadata, test_mode=False, gpu=None, i_need_old=False, verified_super_key_dict=None, task_id=1):
    n_cla_per_tsk = metadata["n_cla_per_tsk"]
    offset1 = int(sum(n_cla_per_tsk[:task_id]))

    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0., 'common_acc':0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.eval()

    with torch.no_grad():
        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            '''
            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None
            '''
            in_buffer = None
            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer=in_buffer, train=False)
                #################
                if verified_super_key_dict is not None:
                    for item_id in range(len(labels)):
                        for key in verified_super_key_dict.keys():
                            if labels[item_id][key] == 1:
                                labels[item_id][verified_super_key_dict[key]] = 1
                #################
                valid_loss += loss * images.shape[0]
            else:
                if config["method"] == "Supp_duq"  or config["method"] == "Supp_cos":
                    if i_need_old:
                        predictions, _, _ = model(images, zero_test=True)
                    else:
                        predictions, _, _ = model(images)
                else:
                    predictions = model(images)

            labels = labels.bool()
            

            valid_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
            valid_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
            valid_metrics['common_acc'] += metrics.common_acc(predictions, labels, offset1) * images.shape[0]
            data_len += images.shape[0]
        valid_loss /= data_len
        valid_metrics['jaccard_sim'] /= data_len
        valid_metrics['modified_jaccard'] /= data_len
        valid_metrics['strict_acc'] /= data_len
        valid_metrics['recall'] /= data_len
        valid_metrics['common_acc'] /= data_len
    return valid_loss, valid_metrics


def task_train(model, buffer, lifelong_datasets, config, metadata, logbook, dist_args=None, verified_super_key_dict=None,super_class_index=None):

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    best_checkpoint = {
        "model_state_dict": deepcopy(model.method_state_dict()),
        "best_modified_jaccard": 0
    }
    best_checkpoint_file = os.path.join(config['logging_path'], "best_checkpoint")
    if config['use_best_model']:
        if config['task_epoch'] > 0 and os.path.exists(best_checkpoint_file):
            if distributed:
                best_checkpoint = torch.load(best_checkpoint_file, map_location=f"cuda:{dist_args['gpu']}")
            else:
                best_checkpoint = torch.load(best_checkpoint_file)

    task_train_data = lifelong_datasets['train']
    if config["method"] == "agem":
        bsm = 0.0
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    else:
        bsm = config["buffer_sampling_multiplier"]
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    
    task_valid_data = lifelong_datasets['intask_valid']
    cur_task_id = task_train_data.cur_task_id

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            task_train_data_with_buffer, num_replicas=dist_args["world_size"], rank=rank)
        
        train_loader = data.DataLoader(
            task_train_data_with_buffer, batch_size=config["single_batch_size"], shuffle=(train_sampler is None),
            num_workers=config["num_workers"], pin_memory=True, sampler=train_sampler
        )

    else:
        train_sampler = None
        train_loader = data.DataLoader(
        task_train_data_with_buffer, batch_size=config["batch_size"], shuffle=(train_sampler is None),
        num_workers=config["num_workers"], pin_memory=True, sampler=train_sampler)
    
    
    valid_loader = data.DataLoader(
        task_valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=True
    )

    if cur_task_id == 0:
        num_epochs = config['epochs_per_task'] * 2
        print_msg(f"Training for {num_epochs} epochs for the first task (double that the other tasks)")
    else:
        num_epochs = config['epochs_per_task']
    print_msg(f"Starting training of task {cur_task_id} epoch {config['task_epoch']} till epoch {num_epochs}")

    epoch_cur = -1
    for epoch in range(config['task_epoch'], num_epochs):
        epoch_cur = epoch_cur + 1
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        log_dict = {}

        if config['method'] == "lucir_duq" or config["method"] == "duq_my":
            if epoch_cur % config["iter_count"] == config["iter_count"] - 1:   # refine stage
                duq_sampler = EachClassSampler_Buffer(task_train_data_with_buffer)
            else:          # single class stage
                duq_sampler = SingleClassSampler_Buffer(task_train_data_with_buffer)
            batch_duq_sampler = EachClassBatchSampler_Buffer(duq_sampler, config["batch_size"], False)
            train_loader = data.DataLoader(task_train_data_with_buffer, num_workers=config["num_workers"], pin_memory=True, batch_sampler=batch_duq_sampler)
        elif config['method'] == "Supp_duq" or config["method"] == "Supp_cos":
            print('now start to make the sampler')
            train_loader = []
            support_sampler = Support_ClassSampler_Buffer(task_train_data_with_buffer)
            batch_support_sampler = EachClassBatchSampler_Buffer(support_sampler, config["batch_size"], False)
            support_train_loader = data.DataLoader(task_train_data_with_buffer, num_workers=config["num_workers"], pin_memory=True, batch_sampler=batch_support_sampler)
            
            
            query_sampler = EachClassSampler_Buffer(task_train_data_with_buffer)
            batch_query_sampler = EachClassBatchSampler_Buffer(query_sampler, config["batch_size"], False)
            query_train_loader = data.DataLoader(task_train_data_with_buffer, num_workers=config["num_workers"], pin_memory=True, batch_sampler=batch_query_sampler)
            
            query_train_loader = data.DataLoader(task_train_data_with_buffer, batch_size=config["batch_size"], shuffle=(train_sampler is None), num_workers=config["num_workers"], pin_memory=True, sampler=train_sampler)
            train_loader.append(query_train_loader)
            train_loader.append(support_train_loader)
            print('now my sampler is complete')
   #TODO:---------------------------------------------------------
        if config['use_cluster_dataset'] and  (epoch_cur % config["inter_cluster"] == config["inter_cluster"] - 1) and cur_task_id <= config["end_cluster_task"]:
            checkpoint_file = config["cluster_checkpoint"]
            
            if config['cluster_model'] == 'resnet18':
                if checkpoint_file == "None":
                    cluster_model = models.resnet18(pretrained=True)
                else:
                    cluster_model = models.resnet18(pretrained=False)
                    cluster_model = cluster_model.load_state_dict(torch.load(config["cluster_checkpoint"]))   
            elif config['cluster_model'] == 'resnet50':
                if checkpoint_file == "None":
                    cluster_model = models.resnet50(pretrained=True)
                else:
                    cluster_model = models.resnet50(pretrained=False)
                    cluster_model = cluster_model.load_state_dict(torch.load(config["cluster_checkpoint"]))
            elif config['cluster_model'] == 'resnet101':
                if checkpoint_file == "None":
                    cluster_model = models.resnet101(pretrained=True)
                else:
                    cluster_model = models.resnet101(pretrained=False)
                    cluster_model = cluster_model.load_state_dict(torch.load(config["cluster_checkpoint"]))
                    
            elif config['cluster_model'] == 'resnet152':
                if checkpoint_file == "None":
                    cluster_model = models.resnet152(pretrained=True)
                else:
                    cluster_model = models.resnet152(pretrained=False)
                    cluster_model = cluster_model.load_state_dict(torch.load(config["cluster_checkpoint"]))
            else:
                raise NotImplementedError
            cluster_model.fc = nn.Identity()
            cluster_model = cluster_model.to(config["device"])
            
            if distributed:
                DDPCluster_DataSet= DDPClusteringDataset(task_train_data_with_buffer.task_data, cluster_model, K_Cluster=config["K_Cluster"], log_book=logbook, rank=rank, batchsize=config['single_batch_size'], world_size=dist_args["world_size"])
                DDPclusterSampler = DDPBaseBatchSampler(DDPCluster_DataSet, num_replicas=dist_args["world_size"], rank=rank, batch_size=config['single_batch_size'], epoch=epoch)
                Cluster_train_loader = data.DataLoader(DDPCluster_DataSet, num_workers=config["num_workers"], pin_memory=True, batch_sampler=DDPclusterSampler)
                dist.barrier() #to calculate the time based on the slowest gpu
                Cluster_epoch_train(model=model, dataloader=Cluster_train_loader, config=config, metadata=metadata, gpu=gpu)
            #    print(f'rank {rank} has finished the cluster training')
                dist.barrier() #to calculate the time based on the slowest gpu

            else:
                Cluster_DataSet = ClusteringDataset(task_train_data_with_buffer.task_data, cluster_model, K_Cluster=config["K_Cluster"], log_book=logbook)
                Cluster_Sampler = SameClassSampler(Cluster_DataSet)
                Cluster_BatcherSampler = SameClassBatcherSampler(Cluster_Sampler, batch_size=config["batch_size"], drop_last=True)
                Cluster_train_loader = data.DataLoader(Cluster_DataSet, num_workers=config["num_workers"], pin_memory=True, batch_sampler=Cluster_BatcherSampler)
                Cluster_epoch_train(model=model, dataloader=Cluster_train_loader, config=config, metadata=metadata, gpu=gpu)
        
        if config['method'] == 'Supp_duq' or config['method'] == "Supp_cos":
            train_loss, train_metrics = Support_epoch_train(model, train_loader, config, metadata, gpu, rank, epoch_cur, cur_task_id=cur_task_id)
        else:
            train_loss, train_metrics = epoch_train(model, train_loader, config, metadata, gpu, rank, epoch_cur) 
            
        
        log_dict[f"train_loss_{cur_task_id}"] = train_loss
        for metric in train_metrics.keys():
            log_dict[f"train_{metric}_{cur_task_id}"] = train_metrics[metric]
            
        
        valid_loss, valid_metrics = evaluate(model, valid_loader, config, metadata, test_mode=False, gpu=gpu, verified_super_key_dict=verified_super_key_dict)
        log_dict[f"valid_loss_{cur_task_id}"] = valid_loss
        for metric in valid_metrics.keys():
            log_dict[f"valid_{metric}_{cur_task_id}"] = valid_metrics[metric]
        

        model.net.eval()
        model.consolidate_epoch_knowledge(log_dict[f"valid_modified_jaccard_{cur_task_id}"],
                                          task_data=task_train_data,
                                          device=config["device"],
                                          batch_size=config["batch_size"])
         
        # If using the lmdb database, close it and open a new environment to kill active readers
        buffer.reset_lmdb_database()

        if config['use_best_model']:
            if log_dict[f"valid_modified_jaccard_{cur_task_id}"] >= best_checkpoint["best_modified_jaccard"]:
                best_checkpoint["best_modified_jaccard"] = log_dict[f"valid_modified_jaccard_{cur_task_id}"]
                best_checkpoint["model_state_dict"] = deepcopy(model.method_state_dict())
            log_dict[f"best_valid_modified_jaccard_{cur_task_id}"] = best_checkpoint["best_modified_jaccard"]

        if distributed:
            dist.barrier() #to calculate the time based on the slowest gpu
        end_time = time.time()
        log_dict[f"elapsed_time"] =  round(end_time - start_time, 2)

        if rank == 0:
        #    utils.log(epoch, cur_task_id, log_dict, logbook)
            if distributed:
                dist.barrier()
        #        log_dict["rank"] = rank
        #        print_msg(log_dict)
                
        else:
            dist.barrier()
            #log_dict["rank"] = rank
            #print_msg(log_dict)
        
        # Checkpointing
        config["task_epoch"] = epoch + 1
        if (config["task_epoch"] % config['checkpoint_interval']) == 0 and rank == 0:
            print_msg("Saving latest checkpoint")
            save_file = os.path.join(config['logging_path'], "latest_model")
            lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)
            if config['use_best_model']:
                print_msg("Saving best checkpoint")
                torch.save(best_checkpoint, best_checkpoint_file)

    # reset the model parameters to the best performing model
    if config['use_best_model']:
        model.load_method_state_dict(best_checkpoint["model_state_dict"])
    
    #--------------------------------------------save the best model of each task ----------------------------------- 
    
    #save_file = os.path.join(config['logging_path'], f'{cur_task_id}_model')
    #lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)
    


def tasks_eval(model, dataset, cur_task_id, config, metadata, logbook, dataset_type="valid", dist_args=None):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """

    i_need_old = False
    if dataset_type == 'test' and cur_task_id == 1:
        i_need_old = False



    assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0
    metrics_dict = {}
    '''
    for task_id in range(cur_task_id + 1):
        dataset.choose_task(task_id)
        dataloader = data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )
        _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu, i_need_old=i_need_old)
        for metric in metrics.keys():
            metrics_dict[f"task_{task_id}_{dataset_type}_{metric}"] = metrics[metric]
    '''


    dataset.load_tasks_up_to(cur_task_id)
    dataloader = data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu, i_need_old=i_need_old, task_id=cur_task_id)
    for metric in metrics.keys():
        metrics_dict[f"average_{dataset_type}_{metric}"] = metrics[metric]

    if rank == 0:
        utils.log_task(cur_task_id, metrics_dict, logbook)
        if distributed:
            dist.barrier()
            metrics_dict["rank"] = rank
            print_msg(metrics_dict)
    else:
        dist.barrier()
        metrics_dict["rank"] = rank
        print_msg(metrics_dict)
