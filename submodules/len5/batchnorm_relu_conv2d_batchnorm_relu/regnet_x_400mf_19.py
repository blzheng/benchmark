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
        self.batchnorm2d62 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d63 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d63 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu59 = ReLU(inplace=True)

    def forward(self, x202):
        x203=self.batchnorm2d62(x202)
        x204=self.relu58(x203)
        x205=self.conv2d63(x204)
        x206=self.batchnorm2d63(x205)
        x207=self.relu59(x206)
        return x207

m = M().eval()
x202 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x202)
end = time.time()
print(end-start)
