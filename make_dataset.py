import os
import shutil
import argparse


def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold, vals[0])
            if not os.path.exists(dest_fold):
                os.mkdir(dest_fold)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))


def create_train_fold(original_fold, data_fold, test_fold):
    # list dirs
    dir_names = list()
    for file in os.listdir(test_fold):
        if os.path.isdir(os.path.join(test_fold, file)):
            dir_names.append(file)

    # build train fold
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(test_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))


def make_dataset(gcommands_fold, out_path):
    validation_path = os.path.join(gcommands_fold, 'validation_list.txt')
    test_path = os.path.join(gcommands_fold, 'testing_list.txt')

    valid_fold = os.path.join(out_path, 'valid')
    test_fold = os.path.join(out_path, 'test')
    train_fold = os.path.join(out_path, 'train')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(valid_fold):
        os.mkdir(valid_fold)
    if not os.path.exists(test_fold):
        os.mkdir(test_fold)
    if not os.path.exists(train_fold):
        os.mkdir(train_fold)

    move_files(gcommands_fold, test_fold, test_path)
    move_files(gcommands_fold, valid_fold, validation_path)
    create_train_fold(gcommands_fold, train_fold, test_fold)


parser = argparse.ArgumentParser(description='Make google commands dataset.')
parser.add_argument('gcommads_fold', help='the path to the root folder of te google commands dataset.')
parser.add_argument('--out_path', default='gcommands', help='the path where to save the files splitted to folders.')

if __name__ == '__main__':
    args = parser.parse_args()
    make_dataset(args.gcommads_fold, args.out_path)
