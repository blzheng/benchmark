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
        self.batchnorm2d9 = BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x54):
        x55=self.batchnorm2d9(x54)
        x56=self.relu6(x55)
        return x56

m = M().eval()
x54 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x54)
end = time.time()
print(end-start)
