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
        self.conv2d67 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x222):
        x223=self.conv2d67(x222)
        x224=self.batchnorm2d67(x223)
        x225=self.relu64(x224)
        return x225

m = M().eval()
x222 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x222)
end = time.time()
print(end-start)
