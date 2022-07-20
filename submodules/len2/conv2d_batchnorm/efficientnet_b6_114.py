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
        self.conv2d192 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x602):
        x603=self.conv2d192(x602)
        x604=self.batchnorm2d114(x603)
        return x604

m = M().eval()
x602 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x602)
end = time.time()
print(end-start)
