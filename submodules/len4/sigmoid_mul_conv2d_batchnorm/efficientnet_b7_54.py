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
        self.sigmoid54 = Sigmoid()
        self.conv2d271 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d161 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x851, x847):
        x852=self.sigmoid54(x851)
        x853=operator.mul(x852, x847)
        x854=self.conv2d271(x853)
        x855=self.batchnorm2d161(x854)
        return x855

m = M().eval()
x851 = torch.randn(torch.Size([1, 3840, 1, 1]))
x847 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x851, x847)
end = time.time()
print(end-start)
