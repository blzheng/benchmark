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
        self.conv2d167 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d99 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x499, x485):
        x500=operator.add(x499, x485)
        x501=self.conv2d167(x500)
        x502=self.batchnorm2d99(x501)
        return x502

m = M().eval()
x499 = torch.randn(torch.Size([1, 224, 14, 14]))
x485 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x499, x485)
end = time.time()
print(end-start)
