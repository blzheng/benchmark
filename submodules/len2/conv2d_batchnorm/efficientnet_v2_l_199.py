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
        self.conv2d309 = Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
        self.batchnorm2d199 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x992):
        x993=self.conv2d309(x992)
        x994=self.batchnorm2d199(x993)
        return x994

m = M().eval()
x992 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x992)
end = time.time()
print(end-start)
