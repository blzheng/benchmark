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
        self.sigmoid16 = Sigmoid()
        self.conv2d83 = Conv2d(720, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x253, x249):
        x254=self.sigmoid16(x253)
        x255=operator.mul(x254, x249)
        x256=self.conv2d83(x255)
        x257=self.batchnorm2d49(x256)
        return x257

m = M().eval()
x253 = torch.randn(torch.Size([1, 720, 1, 1]))
x249 = torch.randn(torch.Size([1, 720, 7, 7]))
start = time.time()
output = m(x253, x249)
end = time.time()
print(end-start)
