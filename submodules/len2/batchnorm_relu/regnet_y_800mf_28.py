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
        self.batchnorm2d45 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)

    def forward(self, x223):
        x224=self.batchnorm2d45(x223)
        x225=self.relu54(x224)
        return x225

m = M().eval()
x223 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x223)
end = time.time()
print(end-start)
