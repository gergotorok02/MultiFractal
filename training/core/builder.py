import torch
import torch.nn as nn
import torch.optim as optim
from common.model import CIFARResNet18

def build_objects(cfg):
    device = torch.device(cfg.device)
    model = CIFARResNet18().to(device)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    return device, model, train_criterion, eval_criterion, opt
