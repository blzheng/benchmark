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
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(224, 224, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d3 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x7):
        x8=self.relu1(x7)
        x9=self.conv2d3(x8)
        x10=self.batchnorm2d3(x9)
        return x10

m = M().eval()
x7 = torch.randn(torch.Size([1, 224, 112, 112]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)
