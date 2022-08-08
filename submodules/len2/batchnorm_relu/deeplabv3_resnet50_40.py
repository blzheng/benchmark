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
        self.batchnorm2d60 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU()

    def forward(self, x204):
        x205=self.batchnorm2d60(x204)
        x206=self.relu56(x205)
        return x206

m = M().eval()
x204 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)