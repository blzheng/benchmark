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
        self.conv2d71 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d72 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x219, x216):
        x220=self.conv2d71(x219)
        x221=self.sigmoid14(x220)
        x222=operator.mul(x221, x216)
        x223=self.conv2d72(x222)
        x224=self.batchnorm2d42(x223)
        return x224

m = M().eval()
x219 = torch.randn(torch.Size([1, 32, 1, 1]))
x216 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x219, x216)
end = time.time()
print(end-start)
