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
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)

    def forward(self, x277, x270):
        x278=self.batchnorm2d84(x277)
        x279=operator.add(x278, x270)
        x280=self.relu79(x279)
        return x280

m = M().eval()
x277 = torch.randn(torch.Size([1, 1024, 14, 14]))
x270 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x277, x270)
end = time.time()
print(end-start)
