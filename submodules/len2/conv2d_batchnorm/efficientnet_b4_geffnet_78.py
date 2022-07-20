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
        self.conv2d130 = Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
        self.batchnorm2d78 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x386):
        x387=self.conv2d130(x386)
        x388=self.batchnorm2d78(x387)
        return x388

m = M().eval()
x386 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x386)
end = time.time()
print(end-start)
