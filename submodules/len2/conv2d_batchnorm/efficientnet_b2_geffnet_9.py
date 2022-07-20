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
        self.conv2d15 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x45):
        x46=self.conv2d15(x45)
        x47=self.batchnorm2d9(x46)
        return x47

m = M().eval()
x45 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
