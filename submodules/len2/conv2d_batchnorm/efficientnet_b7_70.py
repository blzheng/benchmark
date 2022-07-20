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
        self.conv2d118 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d70 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x370):
        x371=self.conv2d118(x370)
        x372=self.batchnorm2d70(x371)
        return x372

m = M().eval()
x370 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x370)
end = time.time()
print(end-start)
