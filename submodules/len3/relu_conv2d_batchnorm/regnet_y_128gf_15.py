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
        self.relu29 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d25 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x121):
        x122=self.relu29(x121)
        x123=self.conv2d39(x122)
        x124=self.batchnorm2d25(x123)
        return x124

m = M().eval()
x121 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
