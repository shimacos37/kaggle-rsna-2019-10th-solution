from __future__ import division, print_function


import torch
import torch.nn as nn


def get_weighted_binary_cross_entropy():
    loss = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]))
    return loss


def get_binary_cross_entropy():
    loss = nn.BCEWithLogitsLoss()
    return loss


def get_any_binary_cross_entropy():
    loss = nn.BCEWithLogitsLoss()
    return loss


def get_weighted_cross_entropy():
    loss = nn.CrossEntropyLoss(reduction="none")
    return loss


def get_loss(loss_name, **params):
    print("loss name:", loss_name)
    f = globals().get("get_" + loss_name)
    return f(**params)
