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
        self.batchnorm2d73 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x251):
        x252=self.batchnorm2d73(x251)
        x253=torch.nn.functional.relu(x252,inplace=True)
        return x253

m = M().eval()
x251 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
