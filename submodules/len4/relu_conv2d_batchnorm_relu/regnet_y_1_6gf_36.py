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
        self.relu77 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d62 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu78 = ReLU(inplace=True)

    def forward(self, x315):
        x316=self.relu77(x315)
        x317=self.conv2d100(x316)
        x318=self.batchnorm2d62(x317)
        x319=self.relu78(x318)
        return x319

m = M().eval()
x315 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)
