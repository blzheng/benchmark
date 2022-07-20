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
        self.conv2d173 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(2064, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x541, x526):
        x542=operator.add(x541, x526)
        x543=self.conv2d173(x542)
        x544=self.batchnorm2d103(x543)
        return x544

m = M().eval()
x541 = torch.randn(torch.Size([1, 344, 7, 7]))
x526 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x541, x526)
end = time.time()
print(end-start)
