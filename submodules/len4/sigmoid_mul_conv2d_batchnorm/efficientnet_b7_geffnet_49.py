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
        self.conv2d246 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d146 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x734, x730):
        x735=x734.sigmoid()
        x736=operator.mul(x730, x735)
        x737=self.conv2d246(x736)
        x738=self.batchnorm2d146(x737)
        return x738

m = M().eval()
x734 = torch.randn(torch.Size([1, 2304, 1, 1]))
x730 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x734, x730)
end = time.time()
print(end-start)
