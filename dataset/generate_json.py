"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/12 下午5:12
@Author  : Yang "Jan" Xiao 
@Description : To generate the json file for disjoint, blurry and several datasets.
"""
import argparse
import json
import os
import random


def readlines(datapath):
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="./", type=str, help="The path of dataset")
    parser.add_argument("--seed", type=int, default=3, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="gsc", help="[gsc]")
    parser.add_argument("--n_tasks", type=int, default=6, help="The number of tasks")
    parser.add_argument("--n_cls_a_task", type=int, default=3, help="The number of class of each task")
    parser.add_argument("--n_init_cls", type=int, default=15, help="The number of classes of initial task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="train", help="[train, test]")
    args = parser.parse_args()
    data_path = args.dpath
    class_list_0 = ["four", "marvin", "on", "sheila", "cat", "five"]
    class_list_1 = ["three", "stop", "go", "dog", "one", "eight"]
    class_list_2 = ["down", "no", "nine", "bed", "wow", "happy"]
    class_list_3 = ["yes", "up", "tree", "seven", "six", "two"]
    class_list_4 = ["off", "house", "zero", "left", "right", "bird"]
    class_list = class_list_0 + class_list_1 + class_list_2 + class_list_3 + class_list_4
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    random.seed(args.seed)
    random.shuffle(class_list)
    total_list = []
    for i in range(args.n_tasks):
        if i == 0:
            t_list = []
            for j in range(args.n_init_cls):
                t_list.append(class_list[j])
            total_list.append(t_list)
        else:
            t_list = []
            for j in range(args.n_cls_a_task):
                t_list.append((class_list[j + args.n_init_cls + (i - 1) * args.n_cls_a_task]))
            total_list.append(t_list)

    print(total_list)
    label_list = []
    for i in range(len(total_list)):
        class_list = total_list[i]
        label_list = label_list + class_list
        if args.mode == 'train':
            collection_name = "collection/{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, mode='train', exp=args.exp_name, rnd=args.seed, n_cls=args.n_cls_a_task, iter=i
            )
            filename = train_filename
        else:
            collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls_a_task, iter=i
            )
            filename = valid_filename
        f = open(collection_name, 'w')
        class_encoding = {category: index for index, category in enumerate(label_list)}
        dataset_list = []
        for path in filename:
            category, wave_name = path.split("/")
            if category in class_list:
                path = os.path.join(category, wave_name)
                dataset_list.append([path, category, class_encoding.get(category)])
        res = [{"klass": item[1], "file_name": item[0], "label": item[2]} for item in dataset_list]

        print("Task ID is {}".format(i))
        print("Total samples are {}".format(len(res)))
        f.write(json.dumps(res))
        f.close()


if __name__ == "__main__":
    main()
