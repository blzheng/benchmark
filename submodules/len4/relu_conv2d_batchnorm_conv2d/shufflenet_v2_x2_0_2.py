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
        self.relu6 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(122, 122, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=122, bias=False)
        self.batchnorm2d10 = BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(122, 122, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x55):
        x56=self.relu6(x55)
        x57=self.conv2d10(x56)
        x58=self.batchnorm2d10(x57)
        x59=self.conv2d11(x58)
        return x59

m = M().eval()
x55 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
