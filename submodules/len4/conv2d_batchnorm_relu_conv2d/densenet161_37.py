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
        self.conv2d77 = Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu78 = ReLU(inplace=True)
        self.conv2d78 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x276):
        x277=self.conv2d77(x276)
        x278=self.batchnorm2d78(x277)
        x279=self.relu78(x278)
        x280=self.conv2d78(x279)
        return x280

m = M().eval()
x276 = torch.randn(torch.Size([1, 1296, 14, 14]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
