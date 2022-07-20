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
        self.batchnorm2d20 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)

    def forward(self, x73):
        x74=self.batchnorm2d20(x73)
        x75=self.relu20(x74)
        return x75

m = M().eval()
x73 = torch.randn(torch.Size([1, 224, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)