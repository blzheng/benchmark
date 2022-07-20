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
        self.batchnorm2d53 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)

    def forward(self, x269):
        x270=self.batchnorm2d53(x269)
        x271=self.relu66(x270)
        return x271

m = M().eval()
x269 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
