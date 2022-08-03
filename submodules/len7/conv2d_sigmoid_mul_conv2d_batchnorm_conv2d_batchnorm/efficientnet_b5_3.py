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
        self.conv2d66 = Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d67 = Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d68 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(768, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x205, x202):
        x206=self.conv2d66(x205)
        x207=self.sigmoid13(x206)
        x208=operator.mul(x207, x202)
        x209=self.conv2d67(x208)
        x210=self.batchnorm2d39(x209)
        x211=self.conv2d68(x210)
        x212=self.batchnorm2d40(x211)
        return x212

m = M().eval()
x205 = torch.randn(torch.Size([1, 16, 1, 1]))
x202 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x205, x202)
end = time.time()
print(end-start)
