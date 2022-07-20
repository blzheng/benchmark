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
        self.batchnorm2d27 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x97):
        x98=self.batchnorm2d27(x97)
        x99=self.relu27(x98)
        x100=self.conv2d27(x99)
        return x100

m = M().eval()
x97 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x97)
end = time.time()
print(end-start)
