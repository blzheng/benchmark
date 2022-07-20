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
        self.batchnorm2d38 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x125, x128):
        x126=self.batchnorm2d38(x125)
        x129=operator.add(x126, x128)
        x130=self.relu34(x129)
        x131=self.conv2d40(x130)
        return x131

m = M().eval()
x125 = torch.randn(torch.Size([1, 1024, 14, 14]))
x128 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x125, x128)
end = time.time()
print(end-start)
