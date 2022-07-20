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
        self.conv2d97 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d98 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x298, x295):
        x299=self.conv2d97(x298)
        x300=self.sigmoid19(x299)
        x301=operator.mul(x300, x295)
        x302=self.conv2d98(x301)
        x303=self.batchnorm2d58(x302)
        return x303

m = M().eval()
x298 = torch.randn(torch.Size([1, 58, 1, 1]))
x295 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x298, x295)
end = time.time()
print(end-start)
