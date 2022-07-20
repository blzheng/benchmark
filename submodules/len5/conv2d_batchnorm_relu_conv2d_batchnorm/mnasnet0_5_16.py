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
        self.conv2d24 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        self.batchnorm2d25 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x68):
        x69=self.conv2d24(x68)
        x70=self.batchnorm2d24(x69)
        x71=self.relu16(x70)
        x72=self.conv2d25(x71)
        x73=self.batchnorm2d25(x72)
        return x73

m = M().eval()
x68 = torch.randn(torch.Size([1, 40, 14, 14]))
start = time.time()
output = m(x68)
end = time.time()
print(end-start)
