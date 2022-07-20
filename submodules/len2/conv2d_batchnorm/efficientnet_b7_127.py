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
        self.conv2d213 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d127 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x670):
        x671=self.conv2d213(x670)
        x672=self.batchnorm2d127(x671)
        return x672

m = M().eval()
x670 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x670)
end = time.time()
print(end-start)
