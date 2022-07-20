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
        self.conv2d67 = Conv2d(896, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d67 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x219, x229):
        x220=self.conv2d67(x219)
        x221=self.batchnorm2d67(x220)
        x230=operator.add(x221, x229)
        return x230

m = M().eval()
x219 = torch.randn(torch.Size([1, 896, 14, 14]))
x229 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x219, x229)
end = time.time()
print(end-start)
