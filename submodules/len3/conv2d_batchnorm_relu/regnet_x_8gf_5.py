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
        self.conv2d9 = Conv2d(80, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)

    def forward(self, x25):
        x28=self.conv2d9(x25)
        x29=self.batchnorm2d9(x28)
        x30=self.relu7(x29)
        return x30

m = M().eval()
x25 = torch.randn(torch.Size([1, 80, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
