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
        self.conv2d169 = Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x538, x523):
        x539=operator.add(x538, x523)
        x540=self.conv2d169(x539)
        return x540

m = M().eval()
x538 = torch.randn(torch.Size([1, 256, 7, 7]))
x523 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x538, x523)
end = time.time()
print(end-start)
