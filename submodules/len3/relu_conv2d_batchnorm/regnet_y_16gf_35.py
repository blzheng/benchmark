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
        self.relu69 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(3024, 3024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=27, bias=False)
        self.batchnorm2d57 = BatchNorm2d(3024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x285):
        x286=self.relu69(x285)
        x287=self.conv2d91(x286)
        x288=self.batchnorm2d57(x287)
        return x288

m = M().eval()
x285 = torch.randn(torch.Size([1, 3024, 14, 14]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
