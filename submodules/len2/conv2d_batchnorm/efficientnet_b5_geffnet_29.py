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
        self.conv2d49 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d29 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x147):
        x148=self.conv2d49(x147)
        x149=self.batchnorm2d29(x148)
        return x149

m = M().eval()
x147 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
