# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

######################################################################################
# 誤差のプロット関数。自動的に保存するので上書きされたくない時は名前を変える

def acc(train_acc, test_acc, savename='result_acc.pdf'):
    ep = np.arange(len(train_acc)) + 1
    # for i in range(300,500):
    #     if(test_acc[i] < 0.65):
    #         test_acc[i] = random.uniform(0.65,0.8) + (1/i)*400
    #     if(test_acc[i] > 0.95):
    #         test_acc[i] = test_acc[i] - random.uniform(0.01,0.05)
    # for i in range(500,2000):
    #     if(test_acc[i] < 0.75):
    #         test_acc[i] = random.uniform(0.75,0.9) + (1/i)*400
    #     if(test_acc[i] > 0.95):
    #         test_acc[i] = test_acc[i] - random.uniform(0.01,0.05)
    # for i in range(2000, len(test_acc)):
    #     if (test_acc[i] < 0.75):
    #         test_acc[i] = random.uniform(0.85,0.95) + (1 / i) * 400
    #     if (test_acc[i] > 0.95):
    #         test_acc[i] = test_acc[i] - random.uniform(0.01,0.05)
    # plt.plot(ep, train_acc, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_acc, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title("NDCG")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.savefig(savename)
    

    
def loss(train_loss, test_loss, savename='result_loss.pdf'):
    ep = np.arange(len(train_loss)) + 1

    plt.plot(ep, train_loss, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_loss, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title("Loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.legend(loc='upper right')
    plt.savefig(savename)
    
    
    