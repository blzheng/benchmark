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
        self.conv2d105 = Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False)
        self.batchnorm2d105 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x344):
        x348=self.conv2d105(x344)
        x349=self.batchnorm2d105(x348)
        return x349

m = M().eval()
x344 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
