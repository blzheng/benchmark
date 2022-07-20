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
        self.conv2d148 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d88 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x443):
        x444=self.conv2d148(x443)
        x445=self.batchnorm2d88(x444)
        return x445

m = M().eval()
x443 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x443)
end = time.time()
print(end-start)
