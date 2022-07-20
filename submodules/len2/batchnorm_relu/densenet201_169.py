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
        self.batchnorm2d169 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu169 = ReLU(inplace=True)

    def forward(self, x597):
        x598=self.batchnorm2d169(x597)
        x599=self.relu169(x598)
        return x599

m = M().eval()
x597 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x597)
end = time.time()
print(end-start)
