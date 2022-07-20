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
        self.batchnorm2d7 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37):
        x38=self.batchnorm2d7(x37)
        x39=torch.nn.functional.relu(x38,inplace=True)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
