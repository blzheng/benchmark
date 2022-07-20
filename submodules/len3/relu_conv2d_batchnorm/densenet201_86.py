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
        self.relu176 = ReLU(inplace=True)
        self.conv2d176 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d177 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x623):
        x624=self.relu176(x623)
        x625=self.conv2d176(x624)
        x626=self.batchnorm2d177(x625)
        return x626

m = M().eval()
x623 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x623)
end = time.time()
print(end-start)
