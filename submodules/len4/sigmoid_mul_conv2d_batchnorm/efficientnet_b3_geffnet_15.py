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
        self.conv2d78 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x229, x225):
        x230=x229.sigmoid()
        x231=operator.mul(x225, x230)
        x232=self.conv2d78(x231)
        x233=self.batchnorm2d46(x232)
        return x233

m = M().eval()
x229 = torch.randn(torch.Size([1, 816, 1, 1]))
x225 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x229, x225)
end = time.time()
print(end-start)
