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
        self.batchnorm2d15 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)

    def forward(self, x72):
        x73=self.batchnorm2d15(x72)
        x74=self.relu17(x73)
        return x74

m = M().eval()
x72 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x72)
end = time.time()
print(end-start)
