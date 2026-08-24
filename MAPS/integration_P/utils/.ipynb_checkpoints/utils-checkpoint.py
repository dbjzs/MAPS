from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import random
import os.path as osp

import torch
import matplotlib.pylab as plt
import seaborn as sns


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    # def __init__(self, fpath=None):
    #     self.console = sys.stdout
    #     self.file = None
    #     if fpath is not None:
    #         mkdir_if_missing(os.path.dirname(fpath))
    #         self.file = open(fpath, 'w')

    def __init__(self, fpath=None):
        self.console = sys.__stdout__  # 使用原始的 stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')


    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    # def close(self):
    #     self.console.close()
    #     if self.file is not None:
    #         self.file.close()
    def close(self):
        # 不要关闭 sys.stdout
        if self.file is not None:
            self.file.close()


def draw_violin_plot_correlation(data, training=False, results='_', task='r2r'):
    plt.figure(dpi = 300)
    plt.rcParams["axes.unicode_minus"] = False

    label = ["pearson", "spearman"]
    font_1 = {"size": 14}

    sns.violinplot(data)
    # plt.xlabel("横坐标标签", font_1)
    # plt.ylabel("纵坐标标签", font_1)
    plt.xticks(ticks = [0, 1], labels = label, fontsize = 11)
    plt.yticks(fontsize = 12)
    if training is True:
        plt.xlabel("Training {}".format(results), font_1)
        plt.savefig('./plots/{}_training.png'.format(task))
    else:
        plt.xlabel("Testing {}".format(results), font_1)
        plt.savefig('./plots/{}_testing.png'.format(task))
    plt.close()
        
  

def draw_violin_plot_rmse(data, training=False, results='_', task='r2r'):
    plt.figure(dpi = 300)
    plt.rcParams["axes.unicode_minus"] = False

    label = ["rmse"]
    font_1 = {"size": 14}

    sns.violinplot(data)
    
    plt.xticks(ticks = [0], labels = label, fontsize = 11)
    plt.yticks(fontsize = 12)
    if training is True:
        plt.xlabel("Training {}".format(results), font_1)
        plt.savefig('./plots/{}_training_rmse.png'.format(task))
    else:
        plt.xlabel("Testing {}".format(results), font_1)
        plt.savefig('./plots/{}_testing_rmse.png'.format(task))
    plt.close()
        


def draw_box_plot_correlation(data, training=False, results='_', save_dir='plots'):
    plt.figure(dpi = 300)
    plt.rcParams["axes.unicode_minus"] = False

    label = ["pearson", "spearman"]
    font_1 = {"size": 14}

    sns.boxplot(data, width=0.5, palette="colorblind")
    sns.stripplot(data=data, color='black', size=3, jitter=True)

    plt.xticks(ticks = [0, 1], labels = label, fontsize = 11)
    plt.yticks(fontsize = 12)
    plt.grid(True)
    if training is True:
        plt.xlabel("Training {}".format(results), font_1)
        plt.savefig('./plots/training.png')
    else:
        plt.xlabel("Testing {}".format(results), font_1)
        plt.savefig('./{}/testing.png'.format(save_dir))
    plt.close()


def draw_box_plot_single(data, training=False, results='_', label=['rmse'], save_dir='plots'):
    plt.figure(dpi = 300)
    plt.rcParams["axes.unicode_minus"] = False

    font_1 = {"size": 14}

    sns.boxplot(data, width=0.5)
    sns.stripplot(data=data, color='black', size=3, jitter=True)

    plt.xticks(ticks = [0], labels = label, fontsize = 11)
    plt.yticks(fontsize = 12)
    plt.grid(True)
    if training is True:
        plt.xlabel("Training {}".format(results), font_1)
        plt.savefig('./plots/training_{}.png'.format(label[0]))
    else:
        plt.xlabel("Testing {}".format(results), font_1)
        plt.savefig('./{}/testing_{}.png'.format(save_dir, label[0]))
    plt.close()

