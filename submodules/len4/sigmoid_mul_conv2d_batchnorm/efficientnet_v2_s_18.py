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
        self.sigmoid18 = Sigmoid()
        self.conv2d113 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x357, x353):
        x358=self.sigmoid18(x357)
        x359=operator.mul(x358, x353)
        x360=self.conv2d113(x359)
        x361=self.batchnorm2d75(x360)
        return x361

m = M().eval()
x357 = torch.randn(torch.Size([1, 1536, 1, 1]))
x353 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x357, x353)
end = time.time()
print(end-start)
