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
        self.batchnorm2d102 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)

    def forward(self, x361):
        x362=self.batchnorm2d102(x361)
        x363=self.relu102(x362)
        return x363

m = M().eval()
x361 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x361)
end = time.time()
print(end-start)
