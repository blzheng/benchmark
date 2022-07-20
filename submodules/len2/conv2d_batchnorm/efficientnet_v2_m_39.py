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
        self.conv2d49 = Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x162):
        x163=self.conv2d49(x162)
        x164=self.batchnorm2d39(x163)
        return x164

m = M().eval()
x162 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x162)
end = time.time()
print(end-start)
