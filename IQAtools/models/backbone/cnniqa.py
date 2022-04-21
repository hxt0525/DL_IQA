# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:45:55 2021

@author: HXT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import Model_List

@Model_List.insert_Module()
class CNNIQA(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQA, self).__init__()
        self.conv1  = nn.Conv2d(3, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 1)

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h  = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = F.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        return q
