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
        self.conv2d63 = Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d63 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x217):
        x218=torch.nn.functional.relu(x217,inplace=True)
        x219=self.conv2d63(x218)
        x220=self.batchnorm2d63(x219)
        x221=torch.nn.functional.relu(x220,inplace=True)
        return x221

m = M().eval()
x217 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x217)
end = time.time()
print(end-start)
