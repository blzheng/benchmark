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
        self.conv2d154 = Conv2d(1200, 1200, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1200, bias=False)
        self.batchnorm2d92 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x460):
        x461=self.conv2d154(x460)
        x462=self.batchnorm2d92(x461)
        return x462

m = M().eval()
x460 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
