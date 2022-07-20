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
        self.conv2d39 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x118, x126):
        x127=self.conv2d39(x118)
        x128=self.batchnorm2d39(x127)
        x129=operator.add(x126, x128)
        x130=self.relu34(x129)
        x131=self.conv2d40(x130)
        return x131

m = M().eval()
x118 = torch.randn(torch.Size([1, 512, 28, 28]))
x126 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x118, x126)
end = time.time()
print(end-start)
