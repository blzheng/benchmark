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
        self.conv2d176 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d177 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu177 = ReLU(inplace=True)
        self.conv2d177 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x624):
        x625=self.conv2d176(x624)
        x626=self.batchnorm2d177(x625)
        x627=self.relu177(x626)
        x628=self.conv2d177(x627)
        return x628

m = M().eval()
x624 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x624)
end = time.time()
print(end-start)
