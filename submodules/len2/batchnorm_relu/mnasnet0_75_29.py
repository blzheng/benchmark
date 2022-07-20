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
        self.batchnorm2d43 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)

    def forward(self, x124):
        x125=self.batchnorm2d43(x124)
        x126=self.relu29(x125)
        return x126

m = M().eval()
x124 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)
