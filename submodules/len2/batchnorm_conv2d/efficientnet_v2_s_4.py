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
        self.batchnorm2d66 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d99 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x314):
        x315=self.batchnorm2d66(x314)
        x316=self.conv2d99(x315)
        return x316

m = M().eval()
x314 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x314)
end = time.time()
print(end-start)
