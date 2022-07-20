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
        self.batchnorm2d125 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu125 = ReLU(inplace=True)

    def forward(self, x442):
        x443=self.batchnorm2d125(x442)
        x444=self.relu125(x443)
        return x444

m = M().eval()
x442 = torch.randn(torch.Size([1, 1632, 14, 14]))
start = time.time()
output = m(x442)
end = time.time()
print(end-start)
