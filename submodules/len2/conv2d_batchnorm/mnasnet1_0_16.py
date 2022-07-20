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
        self.conv2d16 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d16 = BatchNorm2d(120, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x45):
        x46=self.conv2d16(x45)
        x47=self.batchnorm2d16(x46)
        return x47

m = M().eval()
x45 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
