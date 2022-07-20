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
        self.batchnorm2d91 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d92 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x301):
        x302=self.batchnorm2d91(x301)
        x303=self.relu88(x302)
        x304=self.conv2d92(x303)
        x305=self.batchnorm2d92(x304)
        x306=self.relu88(x305)
        return x306

m = M().eval()
x301 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
