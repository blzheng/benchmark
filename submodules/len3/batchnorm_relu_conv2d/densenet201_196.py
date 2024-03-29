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
        self.batchnorm2d197 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu197 = ReLU(inplace=True)
        self.conv2d197 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x695):
        x696=self.batchnorm2d197(x695)
        x697=self.relu197(x696)
        x698=self.conv2d197(x697)
        return x698

m = M().eval()
x695 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x695)
end = time.time()
print(end-start)
