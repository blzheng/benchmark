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
        self.conv2d243 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d155 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x775, x770):
        x776=operator.mul(x775, x770)
        x777=self.conv2d243(x776)
        x778=self.batchnorm2d155(x777)
        return x778

m = M().eval()
x775 = torch.randn(torch.Size([1, 3072, 1, 1]))
x770 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x775, x770)
end = time.time()
print(end-start)
