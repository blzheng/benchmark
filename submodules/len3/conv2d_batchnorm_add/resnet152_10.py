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
        self.conv2d29 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x94, x88):
        x95=self.conv2d29(x94)
        x96=self.batchnorm2d29(x95)
        x97=operator.add(x96, x88)
        return x97

m = M().eval()
x94 = torch.randn(torch.Size([1, 128, 28, 28]))
x88 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x94, x88)
end = time.time()
print(end-start)
