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
        self.batchnorm2d35 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(440, 440, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=55, bias=False)

    def forward(self, x172):
        x173=self.batchnorm2d35(x172)
        x174=self.relu41(x173)
        x175=self.conv2d56(x174)
        return x175

m = M().eval()
x172 = torch.randn(torch.Size([1, 440, 14, 14]))
start = time.time()
output = m(x172)
end = time.time()
print(end-start)
