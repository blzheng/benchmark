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
        self.conv2d29 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87):
        x88=self.conv2d29(x87)
        x89=self.batchnorm2d17(x88)
        return x89

m = M().eval()
x87 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
