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

    def forward(self, x129, x122):
        x130=self.batchnorm2d39(x129)
        x131=operator.add(x130, x122)
        x132=self.relu34(x131)
        return x132

m = M().eval()
x129 = torch.randn(torch.Size([1, 1024, 28, 28]))
x122 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x129, x122)
end = time.time()
print(end-start)
