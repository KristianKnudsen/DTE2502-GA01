import torch

import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):
        # Model predictions
        phos_output = y['phos']
        phoc_output = y['phoc']

        # 
        phos_length = y['phos'].shape[1]
        phoc_length = y['phoc'].shape[1]

        # Actual values
        phos_target, phoc_target = targets.split([phos_length, phoc_length], dim=1)

        # tf.keras.losses.MSE
    	# Apply the loss on PHOS features this is a regression loss
    	# Note: This loss should be applicable to the PHOS part of the 
    	# output which is the first part of the output.
        phos_loss = self.phos_w * F.mse_loss(phos_output, phos_target)

        # Cross entropy loss
        # Apply the loss on PHOC features this is a classification loss
    	# Note: This loss should be applicable to the PHOC part of the 
    	# output which is the later part of the output.
        phoc_loss = self.phoc_w * F.binary_cross_entropy(phoc_output, phoc_target)

        loss = phos_loss + phoc_loss
        return loss
