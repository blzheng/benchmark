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
        self.relu100 = ReLU(inplace=True)
        self.conv2d106 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d106 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x349):
        x350=self.relu100(x349)
        x351=self.conv2d106(x350)
        x352=self.batchnorm2d106(x351)
        return x352

m = M().eval()
x349 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x349)
end = time.time()
print(end-start)
