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
        self.conv2d19 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d19 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x61):
        x62=self.conv2d19(x61)
        x63=self.batchnorm2d19(x62)
        x64=self.relu16(x63)
        x65=self.conv2d20(x64)
        return x65

m = M().eval()
x61 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
