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
        self.batchnorm2d207 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1038):
        x1039=self.batchnorm2d207(x1038)
        return x1039

m = M().eval()
x1038 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1038)
end = time.time()
print(end-start)
