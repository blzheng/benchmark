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
        self.batchnorm2d161 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu161 = ReLU(inplace=True)
        self.conv2d161 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x569):
        x570=self.batchnorm2d161(x569)
        x571=self.relu161(x570)
        x572=self.conv2d161(x571)
        return x572

m = M().eval()
x569 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x569)
end = time.time()
print(end-start)