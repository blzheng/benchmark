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
        self.batchnorm2d29 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x95, x88):
        x96=self.batchnorm2d29(x95)
        x97=operator.add(x96, x88)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 512, 28, 28]))
x88 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x95, x88)
end = time.time()
print(end-start)
