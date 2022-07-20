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
        self.batchnorm2d58 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x207):
        x208=self.batchnorm2d58(x207)
        x209=self.relu58(x208)
        x210=self.conv2d58(x209)
        return x210

m = M().eval()
x207 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
