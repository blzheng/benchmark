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
        self.conv2d119 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1344, bias=False)
        self.batchnorm2d85 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x388):
        x389=self.conv2d119(x388)
        x390=self.batchnorm2d85(x389)
        return x390

m = M().eval()
x388 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)
