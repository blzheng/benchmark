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
        self.conv2d194 = Conv2d(1824, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d195 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu195 = ReLU(inplace=True)

    def forward(self, x687):
        x688=self.conv2d194(x687)
        x689=self.batchnorm2d195(x688)
        x690=self.relu195(x689)
        return x690

m = M().eval()
x687 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x687)
end = time.time()
print(end-start)
