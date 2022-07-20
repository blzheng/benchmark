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
        self.batchnorm2d81 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d82 = Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)

    def forward(self, x278):
        x279=self.batchnorm2d81(x278)
        x280=torch.nn.functional.relu(x279,inplace=True)
        x281=self.conv2d82(x280)
        return x281

m = M().eval()
x278 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)
