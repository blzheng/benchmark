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
        self.conv2d199 = Conv2d(2064, 2064, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2064, bias=False)
        self.batchnorm2d119 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x594):
        x595=self.conv2d199(x594)
        x596=self.batchnorm2d119(x595)
        return x596

m = M().eval()
x594 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x594)
end = time.time()
print(end-start)
