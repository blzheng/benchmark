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
        self.batchnorm2d123 = BatchNorm2d(1600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu123 = ReLU(inplace=True)

    def forward(self, x435):
        x436=self.batchnorm2d123(x435)
        x437=self.relu123(x436)
        return x437

m = M().eval()
x435 = torch.randn(torch.Size([1, 1600, 14, 14]))
start = time.time()
output = m(x435)
end = time.time()
print(end-start)