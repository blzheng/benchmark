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
        self.batchnorm2d3 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU6(inplace=True)
        self.conv2d4 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)

    def forward(self, x9):
        x10=self.batchnorm2d3(x9)
        x11=self.relu62(x10)
        x12=self.conv2d4(x11)
        return x12

m = M().eval()
x9 = torch.randn(torch.Size([1, 96, 112, 112]))
start = time.time()
output = m(x9)
end = time.time()
print(end-start)
