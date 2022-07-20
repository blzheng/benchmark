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
        self.conv2d2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x7):
        x8=self.conv2d2(x7)
        x9=self.batchnorm2d2(x8)
        x10=self.relu1(x9)
        return x10

m = M().eval()
x7 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)
