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
        self.batchnorm2d196 = BatchNorm2d(1856, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu196 = ReLU(inplace=True)

    def forward(self, x692):
        x693=self.batchnorm2d196(x692)
        x694=self.relu196(x693)
        return x694

m = M().eval()
x692 = torch.randn(torch.Size([1, 1856, 7, 7]))
start = time.time()
output = m(x692)
end = time.time()
print(end-start)
