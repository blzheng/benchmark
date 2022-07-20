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
        self.conv2d208 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d124 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x621):
        x622=self.conv2d208(x621)
        x623=self.batchnorm2d124(x622)
        return x623

m = M().eval()
x621 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x621)
end = time.time()
print(end-start)
