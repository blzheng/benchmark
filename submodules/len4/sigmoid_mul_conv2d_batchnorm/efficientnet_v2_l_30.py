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
        self.sigmoid30 = Sigmoid()
        self.conv2d187 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x601, x597):
        x602=self.sigmoid30(x601)
        x603=operator.mul(x602, x597)
        x604=self.conv2d187(x603)
        x605=self.batchnorm2d125(x604)
        return x605

m = M().eval()
x601 = torch.randn(torch.Size([1, 2304, 1, 1]))
x597 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x601, x597)
end = time.time()
print(end-start)
