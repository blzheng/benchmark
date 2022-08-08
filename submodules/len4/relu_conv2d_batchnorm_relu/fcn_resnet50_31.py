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
        self.relu46 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU()

    def forward(self, x173):
        x174=self.relu46(x173)
        x175=self.conv2d53(x174)
        x176=self.batchnorm2d53(x175)
        x177=self.relu49(x176)
        return x177

m = M().eval()
x173 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
