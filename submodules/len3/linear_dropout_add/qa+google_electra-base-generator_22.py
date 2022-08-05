import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.linear70 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)

    def forward(self, x523, x491):
        x524=self.linear70(x523)
        x525=self.dropout35(x524)
        x526=operator.add(x525, x491)
        return x526

m = M().eval()
x523 = torch.randn(torch.Size([1, 384, 256]))
x491 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x523, x491)
end = time.time()
print(end-start)
