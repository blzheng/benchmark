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
        self.conv2d91 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d92 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x298, x295):
        x299=self.conv2d91(x298)
        x300=self.sigmoid11(x299)
        x301=operator.mul(x300, x295)
        x302=self.conv2d92(x301)
        x303=self.batchnorm2d68(x302)
        return x303

m = M().eval()
x298 = torch.randn(torch.Size([1, 56, 1, 1]))
x295 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x298, x295)
end = time.time()
print(end-start)
