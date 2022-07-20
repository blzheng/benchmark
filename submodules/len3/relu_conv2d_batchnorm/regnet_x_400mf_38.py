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
        self.relu38 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x136):
        x137=self.relu38(x136)
        x138=self.conv2d43(x137)
        x139=self.batchnorm2d43(x138)
        return x139

m = M().eval()
x136 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
