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
        self.conv2d90 = Conv2d(1248, 1248, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1248, bias=False)
        self.batchnorm2d54 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x276):
        x277=self.conv2d90(x276)
        x278=self.batchnorm2d54(x277)
        return x278

m = M().eval()
x276 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
