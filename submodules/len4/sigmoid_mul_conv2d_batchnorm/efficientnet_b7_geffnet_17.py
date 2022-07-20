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
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x257, x253):
        x258=x257.sigmoid()
        x259=operator.mul(x253, x258)
        x260=self.conv2d86(x259)
        x261=self.batchnorm2d50(x260)
        return x261

m = M().eval()
x257 = torch.randn(torch.Size([1, 480, 1, 1]))
x253 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x257, x253)
end = time.time()
print(end-start)