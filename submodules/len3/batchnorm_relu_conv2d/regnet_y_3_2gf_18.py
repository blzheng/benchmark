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
        self.batchnorm2d55 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)

    def forward(self, x282):
        x283=self.batchnorm2d55(x282)
        x284=self.relu69(x283)
        x285=self.conv2d90(x284)
        return x285

m = M().eval()
x282 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
