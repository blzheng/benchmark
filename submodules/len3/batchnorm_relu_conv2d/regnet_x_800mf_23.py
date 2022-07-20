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
        self.batchnorm2d38 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(672, 672, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=42, bias=False)

    def forward(self, x122):
        x123=self.batchnorm2d38(x122)
        x124=self.relu34(x123)
        x125=self.conv2d39(x124)
        return x125

m = M().eval()
x122 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
