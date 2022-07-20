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
        self.batchnorm2d114 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x377, x370):
        x378=self.batchnorm2d114(x377)
        x379=operator.add(x378, x370)
        x380=self.relu109(x379)
        x381=self.conv2d115(x380)
        return x381

m = M().eval()
x377 = torch.randn(torch.Size([1, 1024, 14, 14]))
x370 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x377, x370)
end = time.time()
print(end-start)
