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
        self.batchnorm2d11 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x50):
        x51=self.batchnorm2d11(x50)
        x52=torch.nn.functional.relu(x51,inplace=True)
        return x52

m = M().eval()
x50 = torch.randn(torch.Size([1, 32, 25, 25]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
