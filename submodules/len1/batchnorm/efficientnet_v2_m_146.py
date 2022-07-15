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
        self.batchnorm2d146 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x729):
        x730=self.batchnorm2d146(x729)
        return x730

m = M().eval()
x729 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x729)
end = time.time()
print(end-start)
