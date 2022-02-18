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


def readlines(datapath):
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="./", type=str, help="The path of dataset")
    parser.add_argument("--seed", type=int, default=1, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="gsc", help="[gsc]")
    parser.add_argument("--n_tasks", type=int, default=21, help="The number of tasks")
    parser.add_argument("--n_cls", type=int, default=1, help="The number of class of each task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="test", help="[train, test]")
    args = parser.parse_args()
    data_path = args.dpath
    # class_list_6 = ["off", "house"]
    # class_list_7 = ["happy", "zero"]
    # class_list_8 = ["marvin", "no"]
    # class_list_9 = ["tree", "up"]
    # class_list_10 = ["bird", "right"]
    if args.seed == 1 and args.n_cls == 3:
        class_list_0 = ["four", "stop", "go", "dog", "cat", "five", "left", "right", "bird", "tree", "one", "eight",
                        "seven", "six", "two"]
        class_list_1 = ["three", "bed", "up"]
        class_list_2 = ["down", "wow", "happy"]
        class_list_3 = ["yes", "no", "nine"]
        class_list_4 = ["marvin", "on", "sheila"]
        class_list_5 = ["off", "house", "zero"]
    elif args.seed == 1 and args.n_cls == 2:
        class_list_0 = ["four", "stop", "go", "dog", "cat", "five", "left", "seven", "six", "two"]
        class_list_1 = ["three", "bed"]
        class_list_2 = ["down", "wow"]
        class_list_3 = ["yes", "nine"]
        class_list_4 = ["on", "sheila"]
        class_list_5 = ["one", "eight"]
        class_list_6 = ["off", "house"]
        class_list_7 = ["happy", "zero"]
        class_list_8 = ["marvin", "no"]
        class_list_9 = ["tree", "up"]
        class_list_10 = ["bird", "right"]
    elif args.seed == 1 and args.n_cls == 6:
        class_list_0 = ["four", "marvin", "on", "sheila", "cat", "five"]
        class_list_1 = ["three", "stop", "go", "dog", "one", "eight"]
        class_list_2 = ["down", "no", "nine", "bed", "wow", "happy"]
        class_list_3 = ["yes", "up", "tree", "seven", "six", "two"]
        class_list_4 = ["off", "house", "zero", "left", "right", "bird"]
    if args.seed == 1 and args.n_cls == 1:
        class_list_0 = ["four", "stop", "go", "dog", "cat", "five", "left", "seven", "six", "two"]
        class_list_1 = ["three"]
        class_list_2 = ["down"]
        class_list_3 = ["yes"]
        class_list_4 = ["on"]
        class_list_5 = ["one"]
        class_list_6 = ["off"]
        class_list_7 = ["happy"]
        class_list_8 = ["marvin"]
        class_list_9 = ["tree"]
        class_list_10 = ["bird"]
        class_list_11 = ["bed"]
        class_list_12 = ["wow"]
        class_list_13 = ["nine"]
        class_list_14 = ["sheila"]
        class_list_15 = ["eight"]
        class_list_16 = ["house"]
        class_list_17 = ["zero"]
        class_list_18 = ["no"]
        class_list_19 = ["up"]
        class_list_20 = ["right"]


    else:
        class_list_0 = ["seven", "six", "two", "sheila", "cat", "five", "no", "one", "eight", "wow", "marvin", "happy",
                        "up", "tree", "three"]
        class_list_1 = ["stop", "go", "dog"]
        class_list_2 = ["off", "house", "bed"]
        class_list_3 = ["four", "on", "yes"]
        class_list_4 = ["zero", "left", "right"]
        class_list_5 = ["down", "nine", "bird"]
        class_list_6 = ["off"]
        class_list_7 = ["happy"]
        class_list_8 = ["marvin"]
        class_list_9 = ["tree"]
        class_list_10 = ["bird"]
        class_list_11 = ["bed"]
        class_list_12 = ["wow"]
        class_list_13 = ["nine"]
        class_list_14 = ["sheila"]
        class_list_15 = ["eight"]
        class_list_16 = ["house"]
        class_list_17 = ["zero"]
        class_list_18 = ["no"]
        class_list_19 = ["up"]
        class_list_20 = ["right"]
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    total_list = [class_list_0, class_list_1, class_list_2, class_list_3, class_list_4, class_list_5, class_list_6,
                  class_list_7,
                  class_list_8,
                  class_list_9,
                  class_list_10,
                  class_list_11,
                  class_list_12,
                  class_list_13,
                  class_list_14,
                  class_list_15,
                  class_list_16,
                  class_list_17,
                  class_list_18,
                  class_list_19,
                  class_list_20,
                  ]

    label_list = []
    for i in range(len(total_list)):
        class_list = total_list[i]
        label_list = label_list + class_list
        if args.mode == 'train':
            collection_name = "collection/{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, mode='train', exp=args.exp_name, rnd=args.seed, n_cls=args.n_cls, iter=i
            )
            filename = train_filename
        else:
            collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls, iter=i
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
        print(len(res))
        f.write(json.dumps(res))
        f.close()


if __name__ == "__main__":
    main()
