from email.mime import base
import imp
import jsonlines
import matplotlib.pyplot as plt
import pdb
from pathlib import Path
import numpy as np

def get_metric_dict(path):
    metric_dict = {}
    with open(path, "r+") as f:
        for item in jsonlines.Reader(f):
            for metric_name in item.keys():
                if metric_name in metric_dict.keys():
                    metric_dict[metric_name].append(item[metric_name])
                else:
                    metric_dict[metric_name] = []
                    metric_dict[metric_name].append(item[metric_name])
    return metric_dict

def draw_single_picture(line_point, title_name, save_path):
    plt.figure()
    x_index = range(len(line_point))
    plt.plot(x_index, line_point, marker='o')
    plt.title(title_name)
    plt.savefig(save_path+title_name+".png")
    return 0

def draw_mutil_picture(all_metric_file, metric_name, save_dir, experiment_name):
    color_set = ['r', 'c', 'g', 'b', 'y', 'm', 'plum', 'gray', 'salmon', 'sienna', 'orchid', 'k', 'brown', 'coral', 'sienna', 'olive', 'deepskyblue', 'indigo'] * 8
    plt.figure()
    i_index = 0
    for experiment_method in all_metric_file:
        line_point = experiment_method[metric_name]
        x_index = range(len(line_point))
        plt.plot(x_index, line_point, c=color_set[i_index], marker='o')
        i_index = i_index + 1
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.legend(experiment_name, loc='upper right')
    plt.title(metric_name)
    plt.savefig(save_dir+metric_name+".png", dpi=500, bbox_inches='tight')
    plt.close()
    return 0

def draw_highest_data(all_metric_file, metric_name, save_dir, experiment_name):
    color_set = ['r', 'c', 'g', 'b', 'y', 'm', 'plum', 'gray', 'salmon', 'sienna', 'orchid', 'k', 'brown', 'coral', 'sienna', 'olive', 'deepskyblue', 'indigo'] * 8
    plt.figure()
    i_index = 0

    for experiment_method in all_metric_file:
        line_point = []
        for j_index in range(len(experiment_method['average_test_jaccard_sim'])):
            line_point.append(experiment_method[f'task_{j_index}_test_{metric_name}'][0])
        x_index = range(len(line_point))
        plt.plot(x_index, line_point, c=color_set[i_index], marker='o')
        i_index = i_index + 1
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.legend(experiment_name, loc='upper right')
    plt.title(f'highst_{metric_name}')
    plt.savefig(save_dir+metric_name+".png", dpi=500, bbox_inches='tight')
    plt.close()
    return 0
    


def analyze_mutil(task_num, save_dir, base_path_root, experiment_name):
    #experiment_name = ['fine_tune', 'agem', 'icarl_cnn', 'icarl_norm', 'lucir', 'Suppduq', 'cos_model', 'refine']
    all_path = []
    all_metric_file = []
    for method_name in experiment_name:
        base_path = f"{base_path_root}{method_name}/"
        all_path.append(base_path)
    for dir_name in all_path:
        file_name = dir_name + "jsonlogs.jsonl"
        metric_dict = get_metric_dict(path=file_name)
        all_metric_file.append(metric_dict)
    need_show_metric = ['average_test_jaccard_sim','average_test_modified_jaccard', 'average_test_strict_acc', 'average_test_recall']
    for task_id in range(task_num):
        important_metric_name = f'task_{task_id}_test_jaccard_sim' 
        need_show_metric.append(important_metric_name)
        important_metric_name = f'task_{task_id}_test_modified_jaccard' 
        need_show_metric.append(important_metric_name)
        important_metric_name = f'task_{task_id}_test_strict_acc'
        need_show_metric.append(important_metric_name)

    for metric_name in need_show_metric:
        draw_mutil_picture(all_metric_file, metric_name, save_dir, experiment_name)

    metric_type = ['jaccard_sim', 'modified_jaccard', 'strict_acc']
    for temp_name in metric_type:
        draw_highest_data(all_metric_file, temp_name, save_dir, experiment_name)

def analyze_single():
    experiment_name = ['fine_tune', 'lucir', 'Supp_duq_all']
    all_path = []
    all_metric_file = []
    for method_name in experiment_name:
        base_path = f"./../my_result/{method_name}/"
        all_path.append(base_path)
    for dir_name in all_path:
        file_name = dir_name + "jsonlogs.jsonl"
        metric_dict = get_metric_dict(path=file_name)
        all_metric_file.append(metric_dict)

    need_show_metric = ['train_loss_0', 'valid_loss_0', 'train_strict_acc_0', 'train_recall_0', 'valid_strict_acc_0', 'valid_recall_0']

    for metric_name in need_show_metric:
        metric_index = 0
        for cur_dict in all_metric_file:
            draw_single_picture(cur_dict[metric_name], metric_name, all_path[metric_index])
            metric_index = metric_index + 1

def cal_the_mean_and_std(base_dir, task_num):
    root_dir = Path(base_dir)
    file_name = 'jsonlogs.jsonl'
    path_gen = root_dir.glob('*')
    experiment_name = []
    for path_temp in path_gen:
        experiment_name.append(str(path_temp) + '/' + file_name)
 
    all_metric_file = []
    for file_name_iter in experiment_name:
        metric_dict = get_metric_dict(path=file_name_iter)
        all_metric_file.append(metric_dict)

    need_cal_metric = ['average_test_modified_jaccard']
    for task_id in range(task_num):
        important_metric_name = f'task_{task_id}_test_modified_jaccard' 
        need_cal_metric.append(important_metric_name)

    ans_dict = {'metrics': [1, 2]}
    ans_dict[f'highest__test_modified_jaccard_mean'] = []
    ans_dict[f'highest__test_modified_jaccard_std'] = []
    for metric_name in need_cal_metric:
        temp_matrix = []
        for i in range(len(all_metric_file)):
            task_temp = all_metric_file[i][metric_name]
            temp_matrix.append(np.array(task_temp))
        temp_matrix = np.array(temp_matrix)
        pdb.set_trace()
        if metric_name == need_cal_metric[0]:

            ans_dict[f'average_test_modified_jaccard_mean'] = temp_matrix.mean(0)
            ans_dict[f'average_test_modified_jaccard_std'] = temp_matrix.std(0)
        else:
            ans_dict[f'highest__test_modified_jaccard_mean'].append(temp_matrix.mean(0)[0])
            ans_dict[f'highest__test_modified_jaccard_std'].append(temp_matrix.std(0)[0])
    key_name = ['average_test_modified_jaccard', 'highest__test_modified_jaccard']
    print(ans_dict[f'average_test_modified_jaccard_mean'])
    print(ans_dict[f'average_test_modified_jaccard_std'])
    pdb.set_trace()
    return ans_dict, key_name

if __name__ == '__main__':
    '''
    save_dir = './../my_result/compare/'
    base_path_root = './../my_result/'
    experiment_name = ['Suppduq', 'refine', 'refine_1', 'icarl_norm', 'fine_tune']
    #-------------------------------------------------------------------------------
    '''
    
    '''
    save_dir = './../my_result/No_and_have/compare/'
    base_path_root = './../my_result/No_and_have/'
    experiment_name = ['Baseline', 'Baseline + DIV', 'Baseline + Refine', 'Baseline + DIV + Refine']
    analyze_mutil(22, save_dir=save_dir, base_path_root=base_path_root, experiment_name=experiment_name)
    '''

    '''
    base_dir = './../my_result/IIRC_NORM/'
    ans_dict, key_name = cal_the_mean_and_std(base_dir, 10)
    '''

    save_dir = './../my_result/uncertainty/compare/'
    base_path_root = './../my_result/uncertainty/'
    experiment_name = ['label_count&label_mean', 'label_mean&fea_mean', 'label_mean&label_std', 'label_mean_max', 'label_mean&fea_std']
    analyze_mutil(22, save_dir=save_dir, base_path_root=base_path_root, experiment_name=experiment_name)
    