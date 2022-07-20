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
        self.conv2d70 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d71 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x218, x215):
        x219=self.conv2d70(x218)
        x220=self.sigmoid14(x219)
        x221=operator.mul(x220, x215)
        x222=self.conv2d71(x221)
        x223=self.batchnorm2d41(x222)
        return x223

m = M().eval()
x218 = torch.randn(torch.Size([1, 20, 1, 1]))
x215 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x218, x215)
end = time.time()
print(end-start)
