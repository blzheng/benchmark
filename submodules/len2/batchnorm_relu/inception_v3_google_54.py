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
        self.batchnorm2d54 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x190):
        x191=self.batchnorm2d54(x190)
        x192=torch.nn.functional.relu(x191,inplace=True)
        return x192

m = M().eval()
x190 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
