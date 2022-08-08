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
        self.batchnorm2d66 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x219, x212):
        x220=self.batchnorm2d66(x219)
        x221=operator.add(x220, x212)
        x222=self.relu61(x221)
        x223=self.conv2d67(x222)
        x224=self.batchnorm2d67(x223)
        return x224

m = M().eval()
x219 = torch.randn(torch.Size([1, 1024, 28, 28]))
x212 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x219, x212)
end = time.time()
print(end-start)
