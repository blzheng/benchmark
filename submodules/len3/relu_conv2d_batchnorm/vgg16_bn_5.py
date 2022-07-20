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
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d9 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x29):
        x30=self.relu8(x29)
        x31=self.conv2d9(x30)
        x32=self.batchnorm2d9(x31)
        return x32

m = M().eval()
x29 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
