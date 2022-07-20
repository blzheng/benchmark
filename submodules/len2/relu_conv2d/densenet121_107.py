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
        self.relu108 = ReLU(inplace=True)
        self.conv2d108 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x385):
        x386=self.relu108(x385)
        x387=self.conv2d108(x386)
        return x387

m = M().eval()
x385 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x385)
end = time.time()
print(end-start)
