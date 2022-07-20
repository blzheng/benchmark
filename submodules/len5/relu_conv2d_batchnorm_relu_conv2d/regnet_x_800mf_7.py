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
        self.relu15 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)

    def forward(self, x58):
        x59=self.relu15(x58)
        x60=self.conv2d19(x59)
        x61=self.batchnorm2d19(x60)
        x62=self.relu16(x61)
        x63=self.conv2d20(x62)
        return x63

m = M().eval()
x58 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x58)
end = time.time()
print(end-start)
