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
        self.batchnorm2d108 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu103 = ReLU(inplace=True)

    def forward(self, x357, x350):
        x358=self.batchnorm2d108(x357)
        x359=operator.add(x358, x350)
        x360=self.relu103(x359)
        return x360

m = M().eval()
x357 = torch.randn(torch.Size([1, 1024, 14, 14]))
x350 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x357, x350)
end = time.time()
print(end-start)
