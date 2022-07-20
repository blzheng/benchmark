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
        self.conv2d57 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d57 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)

    def forward(self, x184):
        x185=self.conv2d57(x184)
        x186=self.batchnorm2d57(x185)
        x187=self.relu53(x186)
        return x187

m = M().eval()
x184 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
