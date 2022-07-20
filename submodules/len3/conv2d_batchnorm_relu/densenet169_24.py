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
        self.conv2d49 = Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)

    def forward(self, x178):
        x179=self.conv2d49(x178)
        x180=self.batchnorm2d50(x179)
        x181=self.relu50(x180)
        return x181

m = M().eval()
x178 = torch.randn(torch.Size([1, 416, 14, 14]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
