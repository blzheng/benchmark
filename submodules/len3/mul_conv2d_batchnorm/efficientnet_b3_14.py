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
        self.conv2d73 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x222, x217):
        x223=operator.mul(x222, x217)
        x224=self.conv2d73(x223)
        x225=self.batchnorm2d43(x224)
        return x225

m = M().eval()
x222 = torch.randn(torch.Size([1, 816, 1, 1]))
x217 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x222, x217)
end = time.time()
print(end-start)
