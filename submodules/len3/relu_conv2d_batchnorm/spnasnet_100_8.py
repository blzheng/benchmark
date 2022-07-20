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
        self.relu16 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(240, 240, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=240, bias=False)
        self.batchnorm2d25 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x80):
        x81=self.relu16(x80)
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        return x83

m = M().eval()
x80 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
