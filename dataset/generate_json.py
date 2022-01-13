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

from model import readlines


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="./", type=str, help="The path of dataset")
    parser.add_argument("--seed", type=int, default=1, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="gsc", help="[gsc]")
    parser.add_argument("--n_tasks", type=int, default="5", help="The number of tasks")
    parser.add_argument("--n_cls", type=int, default=6, help="The number of class of each task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="test", help="[train, test]")
    args = parser.parse_args()
    data_path = args.dpath

    if args.seed == 1:
        class_list_0 = ["four", "stop", "go", "dog", "cat", "five"]
        class_list_1 = ["three", "bed", "up", "tree", "one", "eight"]
        class_list_2 = ["down", "wow", "happy", "left", "right", "bird"]
        class_list_3 = ["yes", "no", "nine", "seven", "six", "two"]
        class_list_4 = ["marvin", "on", "sheila", "off", "house", "zero"]
    elif args.seed == 2:
        class_list_0 = ["four", "marvin", "on", "sheila", "cat", "five"]
        class_list_1 = ["three", "stop", "go", "dog", "one", "eight"]
        class_list_2 = ["down", "no", "nine", "bed", "wow", "happy"]
        class_list_3 = ["yes", "up", "tree", "seven", "six", "two"]
        class_list_4 = ["off", "house", "zero", "left", "right", "bird"]

    else:
        class_list_0 = ["seven", "six", "two", "sheila", "cat", "five"]
        class_list_1 = ["stop", "go", "dog", "no", "one", "eight"]
        class_list_2 = ["off", "house", "bed", "wow", "marvin", "happy"]
        class_list_3 = ["four", "on", "yes", "up", "tree", "three"]
        class_list_4 = ["zero", "left", "right", "down", "nine", "bird"]
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    total_list = [class_list_0, class_list_1, class_list_2, class_list_3, class_list_4]

    for i in range(len(total_list)):
        class_list = total_list[i]
        if args.mode == 'train':
            collection_name = "collection/{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, mode='train', exp=args.exp_name, rnd=args.seed, n_cls=args.n_cls, iter=i
            )
            filename = train_filename
        else:
            collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
                dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls, iter=i
            )
            filename = valid_filename
        f = open(collection_name, 'w')
        class_encoding = {category: index for index, category in enumerate(class_list)}
        dataset_list = []
        for path in filename:
            category, wave_name = path.split("/")
            if category in class_list:
                path = os.path.join(data_path, category, wave_name)
                dataset_list.append([path, category, class_encoding.get(category)])
        res = [{"klass": item[1], "file_name": item[0], "label": item[2]} for item in dataset_list]
        print(len(res))
        f.write(json.dumps(res))
        f.close()


if __name__ == "__main__":
    main()
