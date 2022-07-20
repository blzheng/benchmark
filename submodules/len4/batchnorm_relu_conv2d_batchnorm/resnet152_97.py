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
        self.batchnorm2d150 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu145 = ReLU(inplace=True)
        self.conv2d151 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d151 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x496):
        x497=self.batchnorm2d150(x496)
        x498=self.relu145(x497)
        x499=self.conv2d151(x498)
        x500=self.batchnorm2d151(x499)
        return x500

m = M().eval()
x496 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x496)
end = time.time()
print(end-start)
