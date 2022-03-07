## Rainbow Keywords - Official PyTorch Implementation

## Abstract
Due to the Deep Learning expanding rapidly, it is very attractive to deploy the deep keyword spotting(KWS) paradigm on edge devices. However, the keyword spotting Model deployed on the edge will front a dynamic and resource-constrained environment. In practice, a keyword spotting system should be capable of incrementally learning about unknown classes in the real world with restrained memory resources. We concentrate on this "online class incremental learning" issue. Current keyword spotting approaches are severely limited due to the constrained nature of available resources on the edge devices and the catastrophic forgetting (CF) issue. To alleviate such an issue, we propose a novel diversity-aware class increment learning method Rainbow Keywords(RK) for effectively managing the memory with limited capacity by leveraging classification uncertainty. With extensive empirical validations on the Google Speech Commands dataset, we demonstrate that the proposed method significantly enhances the accuracy in online class incremental learning setups, exceeding other class incremental learning methods by extensive margins(8.6\%), especially beneath lower memory size circumstances.

## Overview of the results of Rainbow Keywords
Here is a list of continual learning methods available for KWS:
- Elastic Weight Consolidation (EWC) [[view]](./methods/regularization.py)
- Riemannian Walk (RWalk) [[view]](./methods/regularization.py)
- Incremental Classifier and Representation Learning(iCaRL) [[view]](./methods/icarl.py)

The table is shown for last accuracy,last forgetting,and intransigence comparison in Google Speech Command datasets (30 classes GSC split 6 tasks, task 1 has 15 classes)-Online with memory size 500.
If you want to see more details, see the paper.

| Methods   | Accuracy     | Forgetting   | Intransigence |
|-----------|------------|------------|----------|
| EWC       | 0.690 | 0.112 | 0.250    |
| Rwalk     | 0.702 | 0.112 | 0.238    |
| iCaRL     | 0.636 | 0.290 | 0.304    |
| Native Rehearsal     | 0.107 | 0.211 | 0.832 |
| BiC       | 0.713 | 0.093 | 0.227    |
| **RK w/o DA** | 0.757 | 0.072 | 0.182 |
| **RK**        |**0.799**|**0.044**|**0.140**|

## Getting Started
### Requirements 
- Python3
- audiomentations==0.20.0
- einops==0.3.2
- matplotlib==3.4.3
- numpy==1.20.3
- pandas==1.3.4
- seaborn==0.11.2
- torch==1.10.0
- torch_optimizer==0.3.0
- torchaudio==0.10.0

### Setup Environment

You need to create the running environment by [Anaconda](https://www.anaconda.com/),

```bash
conda env create -f environment.yml
conda active kws
```

If you don't have Anaconda, you can install (Linux version) by

```bash
cd <ROOT>
bash conda_install.sh
```
### Download Dataset

We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```bash
cd <ROOT>/dataset
bash download_gsc.sh
```

### Usage 
To run the experiments in the paper, you just run `experiment.sh`.
```angular2html
bash experiment.sh 
```
For various experiments, you should know the role of each argument. 

- `MODE`: CIL methods. Our method is called `rainbow_keywords`. [finetune, native_rehearsal, joint, rwalk, icarl, rainbow_keywords, ewc, bic] (`joint` calculates accuracy when training all the datasets at once.)
- `MEM_MANAGE`: Memory management method. `default` uses the memory method which the paper originally used.
  [default, random, reservoir, uncertainty, prototype].
- `RND_SEED`: Random Seed Number 
- `DATASET`: Dataset name [gsc]
- `STREAM`: The setting whether current task data can be seen iteratively or not.[online]                                        
- `MEM_SIZE`: Memory size gsc: k={300, 500, 1000, 1500}
- `TRANS`: Augmentation. Multiple choices [mixup,specaugment]

### Results
There are three types of logs during running experiments; logs, results, tensorboard. 
The log files are saved in `logs` directory, and the results which contains accuracy of each task 
are saved in `results` directory. 
```angular2html
root_directory
    |_ logs 
        |_ [dataset]
            |_.log
            |_ ...
    |_ results
        |_ [dataset]
            |_.npy
            |_...
```

In addition, you can also use the `tensorboard` as following command.
```angular2html
tensorboard --logdir tensorboard
```

## Acknowledgements
Our implementation refers the source code from the following repositories:

- [rainbow-memory](https://github.com/clovaai/rainbow-memory)
- [online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
- [icarl](https://github.com/donlee90/icarl)
