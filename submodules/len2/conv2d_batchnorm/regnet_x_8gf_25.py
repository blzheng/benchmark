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
        self.conv2d25 = Conv2d(240, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x77):
        x80=self.conv2d25(x77)
        x81=self.batchnorm2d25(x80)
        return x81

m = M().eval()
x77 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x77)
end = time.time()
print(end-start)
