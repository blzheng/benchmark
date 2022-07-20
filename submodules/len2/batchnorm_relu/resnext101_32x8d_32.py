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
        self.batchnorm2d50 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)

    def forward(self, x164):
        x165=self.batchnorm2d50(x164)
        x166=self.relu46(x165)
        return x166

m = M().eval()
x164 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
