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
        self.conv2d74 = Conv2d(432, 432, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=432, bias=False)
        self.batchnorm2d44 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x222):
        x223=self.conv2d74(x222)
        x224=self.batchnorm2d44(x223)
        return x224

m = M().eval()
x222 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x222)
end = time.time()
print(end-start)
