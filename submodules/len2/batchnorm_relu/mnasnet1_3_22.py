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
        self.batchnorm2d33 = BatchNorm2d(768, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x95):
        x96=self.batchnorm2d33(x95)
        x97=self.relu22(x96)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x95)
end = time.time()
print(end-start)
