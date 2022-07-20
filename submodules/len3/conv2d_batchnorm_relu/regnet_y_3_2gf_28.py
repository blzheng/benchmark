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
        self.conv2d70 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d44 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)

    def forward(self, x220):
        x221=self.conv2d70(x220)
        x222=self.batchnorm2d44(x221)
        x223=self.relu54(x222)
        return x223

m = M().eval()
x220 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
