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
        self.conv2d26 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d26 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x84):
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        return x86

m = M().eval()
x84 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x84)
end = time.time()
print(end-start)
