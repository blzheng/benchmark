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
        self.conv2d188 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d112 = BatchNorm2d(3072, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x588):
        x589=self.conv2d188(x588)
        x590=self.batchnorm2d112(x589)
        return x590

m = M().eval()
x588 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x588)
end = time.time()
print(end-start)
