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
        self.conv2d25 = Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=320, bias=False)
        self.batchnorm2d25 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87):
        x88=self.conv2d25(x87)
        x89=self.batchnorm2d25(x88)
        return x89

m = M().eval()
x87 = torch.randn(torch.Size([1, 320, 28, 28]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
