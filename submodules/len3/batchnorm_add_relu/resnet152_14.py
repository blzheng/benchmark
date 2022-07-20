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
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x127, x126):
        x128=self.batchnorm2d39(x127)
        x129=operator.add(x126, x128)
        x130=self.relu34(x129)
        return x130

m = M().eval()
x127 = torch.randn(torch.Size([1, 1024, 14, 14]))
x126 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x127, x126)
end = time.time()
print(end-start)
