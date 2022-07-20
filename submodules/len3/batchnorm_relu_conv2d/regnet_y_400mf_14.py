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
        self.batchnorm2d44 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)

    def forward(self, x220):
        x221=self.batchnorm2d44(x220)
        x222=self.relu53(x221)
        x223=self.conv2d71(x222)
        return x223

m = M().eval()
x220 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
