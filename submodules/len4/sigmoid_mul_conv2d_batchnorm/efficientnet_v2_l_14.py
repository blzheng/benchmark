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
        self.sigmoid14 = Sigmoid()
        self.conv2d107 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x347, x343):
        x348=self.sigmoid14(x347)
        x349=operator.mul(x348, x343)
        x350=self.conv2d107(x349)
        x351=self.batchnorm2d77(x350)
        return x351

m = M().eval()
x347 = torch.randn(torch.Size([1, 1344, 1, 1]))
x343 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x347, x343)
end = time.time()
print(end-start)
