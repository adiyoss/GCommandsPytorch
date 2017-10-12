import os
import shutil

validation_path = 'data/validation_list.txt'
test_path = 'data/testing_list.txt'
parent_dir = 'gcommands'
original_fold = 'data'
valid_fold = os.path.join(parent_dir, 'valid')
test_fold = os.path.join(parent_dir, 'test')
train_fold = os.path.join(parent_dir, 'train')

if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
if not os.path.exists(valid_fold):
    os.mkdir(valid_fold)
if not os.path.exists(test_fold):
    os.mkdir(test_fold)
if not os.path.exists(train_fold):
    os.mkdir(train_fold)


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


move_files(original_fold, test_fold, test_path)
move_files(original_fold, valid_fold, validation_path)
create_train_fold(original_fold, train_fold, test_fold)


