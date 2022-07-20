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
        self.conv2d82 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x253, x248):
        x254=operator.mul(x253, x248)
        x255=self.conv2d82(x254)
        x256=self.batchnorm2d48(x255)
        return x256

m = M().eval()
x253 = torch.randn(torch.Size([1, 768, 1, 1]))
x248 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x253, x248)
end = time.time()
print(end-start)
