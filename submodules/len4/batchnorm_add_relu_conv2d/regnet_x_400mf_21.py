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
        self.batchnorm2d58 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x188, x181):
        x189=self.batchnorm2d58(x188)
        x190=operator.add(x181, x189)
        x191=self.relu54(x190)
        x192=self.conv2d59(x191)
        return x192

m = M().eval()
x188 = torch.randn(torch.Size([1, 400, 7, 7]))
x181 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x188, x181)
end = time.time()
print(end-start)
