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
        self.conv2d66 = Conv2d(784, 784, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=49, bias=False)
        self.batchnorm2d42 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)

    def forward(self, x206):
        x207=self.conv2d66(x206)
        x208=self.batchnorm2d42(x207)
        x209=self.relu50(x208)
        return x209

m = M().eval()
x206 = torch.randn(torch.Size([1, 784, 14, 14]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
