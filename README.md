## Rainbow Keywords - Official PyTorch Implementation

## Abstract
Catastrophic forgetting is a thorny challenge when updating keyword spotting (KWS) models after deployment. This problem will be more challenging if KWS models are further required for edge devices due to their limited memory. To alleviate such an issue, we propose a novel diversity-aware incremental learning method named Rainbow Keywords (RK). Specifically, the proposed RK approach introduces a diversity-aware sampler to select a diverse set from historical and incoming keywords by calculating classification uncertainty. As a result, the RK approach can incrementally learn new tasks without forgetting prior knowledge. Besides, the RK approach also proposes data augmentation and knowledge distillation loss function for efficient memory management on the edge device. Experimental results show that the proposed RK approach achieves 4.2% absolute improvement in terms of average accuracy over the best baseline on Google Speech Command dataset with less required memory.

## Overview of the results of Rainbow Keywords
Here is a list of continual learning methods available for KWS:
- Elastic Weight Consolidation (EWC) [[view]](./methods/regularization.py)
- Riemannian Walk (RWalk) [[view]](./methods/regularization.py)
- Incremental Classifier and Representation Learning(iCaRL) [[view]](./methods/icarl.py)
- Bias Correction(BiC) [[view]](./methods/bic.py)
- Rainbow Keywords(RK) [[view]](./methods/)

If you want to see more details, see the paper.

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
To run the experiments in the paper, you just run `run.sh`.
```angular2html
bash run.sh 
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
## Citation
Please cite our paper if it is helpful to your work:
```tex
@article{pclkws,
  author    = {Yizheng Huang and
               Nana Hou and
               Nancy F. Chen},
  title     = {Progressive Continual Learning for Spoken Keyword Spotting},
  journal   = {CoRR},
  volume    = {abs/2201.12546},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.12546},
  eprinttype = {arXiv},
  eprint    = {2201.12546},
  timestamp = {Wed, 02 Feb 2022 15:00:01 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-12546.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@article{rk,
  author    = {Yang Xiao and
               Nana Hou and
               Eng Siong Chng},
  title     = {Rainbow Keywords: Efficient Incremental Learning for Online Spoken
               Keyword Spotting},
  journal   = {CoRR},
  volume    = {abs/2203.16361},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2203.16361},
  doi       = {10.48550/arXiv.2203.16361},
  eprinttype = {arXiv},
  eprint    = {2203.16361},
  timestamp = {Mon, 04 Apr 2022 18:01:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2203-16361.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledgements
Our implementations use the source code from the following repositories and users:
- [rainbow-memory](https://github.com/clovaai/rainbow-memory)
- [online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
- [icarl](https://github.com/donlee90/icarl)
- [@huangyz0918](https://github.com/huangyz0918)


