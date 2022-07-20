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
        self.conv2d143 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d85 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x448):
        x449=self.conv2d143(x448)
        x450=self.batchnorm2d85(x449)
        return x450

m = M().eval()
x448 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x448)
end = time.time()
print(end-start)
