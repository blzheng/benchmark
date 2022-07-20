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
        self.batchnorm2d82 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu104 = ReLU(inplace=True)

    def forward(self, x424, x411):
        x425=self.batchnorm2d82(x424)
        x426=operator.add(x411, x425)
        x427=self.relu104(x426)
        return x427

m = M().eval()
x424 = torch.randn(torch.Size([1, 888, 7, 7]))
x411 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x424, x411)
end = time.time()
print(end-start)
