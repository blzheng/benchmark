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
        self.conv2d132 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x414, x399):
        x415=operator.add(x414, x399)
        x416=self.conv2d132(x415)
        x417=self.batchnorm2d78(x416)
        return x417

m = M().eval()
x414 = torch.randn(torch.Size([1, 160, 14, 14]))
x399 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x414, x399)
end = time.time()
print(end-start)
