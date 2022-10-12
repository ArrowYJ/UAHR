# Uncertainty-Aware Hierarchical Refinement for Incremental Implicitly-Refined Classification
This is the implementation of the UAHR.
IIRC is a setup and benchmark to evaluate lifelong learning models in more dynamic and real-life aligned scenarios, where the labels are changing in a dynamic way and the models have to use what they have learnt to incorporate these changes into their knowledge.

It contains the following two packages as well: **iirc** and **lifelong_methods** 

**iirc** is a package for adapting the different datasets (currently supports *CIFAR-100* and *ImageNet*) to 
the *iirc* setup and the *class incremental learning* setup, and loading them in a standardized manner.

**lifelong_methods** is a package that standardizes the different stages any lifelong learning method passes by, 
hence it provides a faster way for implementing new ideas and embedding them in the same training code as other 
baselines, it provides as well the implementation of some of these baselines. 

## Running Instructions
The starting point for running this code is experiments/main.py.  You can use "sh train_UAHR.sh" to run it.

Or, for example, for reproducing the results of experience replay on iirc-cifar with a buffer of 20 samples per class:
```buildoutcfg
CUDA_VISIBLE_DEVICES=0  python main.py --l_divide 10.0  --l_margin 0.1 --temperature 2.5 --test_incre 0 --dataset iirc_cifar100 --dataset_path "./data" --epochs_per_task 140 --batch_size 128 --seed 100 --logging_path_root "./results" --n_layers 32 --tasks_configuration_id 0 --method UAHR --optimizer momentum --lr 1.0 --lr_gamma 0.1 --reduce_lr_on_plateau --weight_decay 1e-5 --n_memories_per_class 20 --checkpoint_interval 5 
```
### Requirements
This code has been tested with python 3.8.2 and the following packages:

pytorch==1.5.0
<br/>
torchvision==0.6.0
<br/>
numpy==1.18.5
<br/>
Pillow==7.0.0
<br/>
lmdb==1.0.0
<br/>
pip install "mllogger[all]"

<br/>seaborn==0.10.1 (optional)
<br/>
pytest==5.4.3 (optional)
<br/>
pytest-cov==2.9.0 (optional)
<br/>

### CIFAR-100
To be able to run the code with *CIFAR-100* derived datasets, just download the dataset from the 
[official website](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it, or use the 
*iirc/utils/download_cifar.py* file. 

The path that should be provided in the *dataset_path* argument, when running 
*main.py*, is the path of the parent directory of the extracted *cifar-100-python* folder.
