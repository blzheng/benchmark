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
        self.batchnorm2d176 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu176 = ReLU(inplace=True)

    def forward(self, x622):
        x623=self.batchnorm2d176(x622)
        x624=self.relu176(x623)
        return x624

m = M().eval()
x622 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x622)
end = time.time()
print(end-start)
