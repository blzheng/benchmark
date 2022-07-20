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
        self.conv2d227 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid40 = Sigmoid()
        self.conv2d228 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x725, x722):
        x726=self.conv2d227(x725)
        x727=self.sigmoid40(x726)
        x728=operator.mul(x727, x722)
        x729=self.conv2d228(x728)
        return x729

m = M().eval()
x725 = torch.randn(torch.Size([1, 128, 1, 1]))
x722 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x725, x722)
end = time.time()
print(end-start)
