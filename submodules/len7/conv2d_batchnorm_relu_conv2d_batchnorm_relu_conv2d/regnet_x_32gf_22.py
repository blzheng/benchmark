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
        self.conv2d71 = Conv2d(1344, 2520, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(2520, 2520, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=15, bias=False)
        self.batchnorm2d72 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(2520, 2520, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x229):
        x232=self.conv2d71(x229)
        x233=self.batchnorm2d71(x232)
        x234=self.relu67(x233)
        x235=self.conv2d72(x234)
        x236=self.batchnorm2d72(x235)
        x237=self.relu68(x236)
        x238=self.conv2d73(x237)
        return x238

m = M().eval()
x229 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x229)
end = time.time()
print(end-start)
