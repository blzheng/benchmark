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
        self.batchnorm2d120 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu120 = ReLU(inplace=True)

    def forward(self, x424):
        x425=self.batchnorm2d120(x424)
        x426=self.relu120(x425)
        return x426

m = M().eval()
x424 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x424)
end = time.time()
print(end-start)
