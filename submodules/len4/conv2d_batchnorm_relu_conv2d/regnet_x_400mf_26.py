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
        self.conv2d42 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d42 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x134):
        x135=self.conv2d42(x134)
        x136=self.batchnorm2d42(x135)
        x137=self.relu38(x136)
        x138=self.conv2d43(x137)
        return x138

m = M().eval()
x134 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x134)
end = time.time()
print(end-start)
