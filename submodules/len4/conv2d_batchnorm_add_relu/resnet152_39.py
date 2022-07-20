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
        self.conv2d114 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)

    def forward(self, x376, x370):
        x377=self.conv2d114(x376)
        x378=self.batchnorm2d114(x377)
        x379=operator.add(x378, x370)
        x380=self.relu109(x379)
        return x380

m = M().eval()
x376 = torch.randn(torch.Size([1, 256, 14, 14]))
x370 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x376, x370)
end = time.time()
print(end-start)
