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
        self.batchnorm2d10 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x46):
        x47=self.batchnorm2d10(x46)
        x48=torch.nn.functional.relu(x47,inplace=True)
        return x48

m = M().eval()
x46 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
