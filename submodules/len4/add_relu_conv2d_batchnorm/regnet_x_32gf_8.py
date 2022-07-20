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
        self.relu27 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(672, 1344, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87, x95):
        x96=operator.add(x87, x95)
        x97=self.relu27(x96)
        x98=self.conv2d30(x97)
        x99=self.batchnorm2d30(x98)
        return x99

m = M().eval()
x87 = torch.randn(torch.Size([1, 672, 28, 28]))
x95 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x87, x95)
end = time.time()
print(end-start)
