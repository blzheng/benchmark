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
        self.conv2d85 = Conv2d(896, 2016, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(2016, 2016, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=36, bias=False)

    def forward(self, x265):
        x268=self.conv2d85(x265)
        x269=self.batchnorm2d53(x268)
        x270=self.relu65(x269)
        x271=self.conv2d86(x270)
        return x271

m = M().eval()
x265 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
