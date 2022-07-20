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
        self.conv2d59 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d35 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x177):
        x178=self.conv2d59(x177)
        x179=self.batchnorm2d35(x178)
        return x179

m = M().eval()
x177 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
