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
        self.conv2d71 = Conv2d(784, 784, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=49, bias=False)
        self.batchnorm2d45 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x222):
        x223=self.conv2d71(x222)
        x224=self.batchnorm2d45(x223)
        return x224

m = M().eval()
x222 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x222)
end = time.time()
print(end-start)
