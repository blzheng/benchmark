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
        self.conv2d32 = Conv2d(144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d22 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x92):
        x93=self.conv2d32(x92)
        x94=self.batchnorm2d22(x93)
        return x94

m = M().eval()
x92 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
