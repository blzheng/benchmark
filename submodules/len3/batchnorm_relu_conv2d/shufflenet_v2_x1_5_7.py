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
        self.batchnorm2d26 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(176, 176, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=176, bias=False)

    def forward(self, x166):
        x167=self.batchnorm2d26(x166)
        x168=self.relu17(x167)
        x169=self.conv2d27(x168)
        return x169

m = M().eval()
x166 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
