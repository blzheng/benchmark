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
        self.batchnorm2d83 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d142 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x444):
        x445=self.batchnorm2d83(x444)
        x446=self.conv2d142(x445)
        return x446

m = M().eval()
x444 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)
