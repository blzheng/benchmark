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
        self.batchnorm2d15 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x52, x48):
        x53=self.batchnorm2d15(x52)
        x54=operator.add(x53, x48)
        x55=self.relu13(x54)
        return x55

m = M().eval()
x52 = torch.randn(torch.Size([1, 128, 28, 28]))
x48 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x52, x48)
end = time.time()
print(end-start)
