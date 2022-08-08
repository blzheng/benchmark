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
        self.conv2d55 = Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
        self.batchnorm2d55 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x174):
        x181=self.conv2d55(x174)
        x182=self.batchnorm2d55(x181)
        return x182

m = M().eval()
x174 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
