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
        self.batchnorm2d24 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x120):
        x121=self.batchnorm2d24(x120)
        x122=self.relu29(x121)
        x123=self.conv2d39(x122)
        return x123

m = M().eval()
x120 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)