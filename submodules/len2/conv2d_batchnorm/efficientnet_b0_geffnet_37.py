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
        self.conv2d61 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d37 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x178):
        x179=self.conv2d61(x178)
        x180=self.batchnorm2d37(x179)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
