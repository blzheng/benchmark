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
        self.conv2d72 = Conv2d(440, 110, kernel_size=(1, 1), stride=(1, 1))
        self.relu55 = ReLU()
        self.conv2d73 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d74 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x226, x225, x219):
        x227=self.conv2d72(x226)
        x228=self.relu55(x227)
        x229=self.conv2d73(x228)
        x230=self.sigmoid13(x229)
        x231=operator.mul(x230, x225)
        x232=self.conv2d74(x231)
        x233=self.batchnorm2d46(x232)
        x234=operator.add(x219, x233)
        return x234

m = M().eval()
x226 = torch.randn(torch.Size([1, 440, 1, 1]))
x225 = torch.randn(torch.Size([1, 440, 7, 7]))
x219 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x226, x225, x219)
end = time.time()
print(end-start)
