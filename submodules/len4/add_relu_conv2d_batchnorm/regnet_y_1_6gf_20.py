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
        self.relu84 = ReLU(inplace=True)
        self.conv2d109 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x329, x343):
        x344=operator.add(x329, x343)
        x345=self.relu84(x344)
        x346=self.conv2d109(x345)
        x347=self.batchnorm2d67(x346)
        return x347

m = M().eval()
x329 = torch.randn(torch.Size([1, 336, 14, 14]))
x343 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x329, x343)
end = time.time()
print(end-start)
