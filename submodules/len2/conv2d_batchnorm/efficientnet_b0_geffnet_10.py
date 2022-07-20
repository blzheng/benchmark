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
        self.conv2d16 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x47):
        x48=self.conv2d16(x47)
        x49=self.batchnorm2d10(x48)
        return x49

m = M().eval()
x47 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
