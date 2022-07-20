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
        self.relu63 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x211, x219):
        x220=operator.add(x211, x219)
        x221=self.relu63(x220)
        x222=self.conv2d68(x221)
        x223=self.batchnorm2d68(x222)
        return x223

m = M().eval()
x211 = torch.randn(torch.Size([1, 400, 7, 7]))
x219 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x211, x219)
end = time.time()
print(end-start)