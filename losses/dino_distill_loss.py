import torch
import torch.nn.functional as F
import torch.nn as nn


def dino_cls_distill_loss(student_outputs, teacher_outputs, loss_type='mse'):
    """
    CLS-level distillation loss between student and teacher outputs.

    Args:
        student_outputs: Tensor or list of tensors (B, C)
        teacher_outputs: Tensor or list of tensors (B, C)
        loss_type: 'mse' or 'cosine'

    Returns:
        Distillation loss (scalar tensor)
    """
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'cosine':
        loss_fn = nn.CosineEmbeddingLoss()
    else:
        raise ValueError("Unsupported loss type: choose 'mse' or 'cosine'")

    if isinstance(student_outputs, list):
        loss = 0
        for s, t in zip(student_outputs, teacher_outputs):
            if loss_type == 'cosine':
                # CosineEmbeddingLoss requires an additional 'target' tensor of 1s
                target = torch.ones(s.size(0)).to(s.device)
                loss += loss_fn(s, t.detach(), target)
            else:
                loss += loss_fn(s, t.detach())
        loss = loss / len(student_outputs)
    else:
        if loss_type == 'cosine':
            target = torch.ones(student_outputs.size(0)).to(student_outputs.device)
            loss = loss_fn(student_outputs, teacher_outputs.detach(), target)
        else:
            loss = loss_fn(student_outputs, teacher_outputs.detach())
    return loss


class DINODistillLoss(nn.Module):
    """
    PyTorch nn.Module version of DINO-style distillation loss for plug-and-play
    """
    def __init__(self, loss_type='mse'):
        super(DINODistillLoss, self).__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
            self.loss_type = 'mse'
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
            self.loss_type = 'cosine'
        else:
            raise ValueError("Unsupported loss type: choose 'mse' or 'cosine'")

    def forward(self, student_outputs, teacher_outputs):
        if isinstance(student_outputs, list):
            loss = 0
            for s, t in zip(student_outputs, teacher_outputs):
                if self.loss_type == 'cosine':
                    target = torch.ones(s.size(0)).to(s.device)
                    loss += self.loss_fn(s, t.detach(), target)
                else:
                    loss += self.loss_fn(s, t.detach())
            loss = loss / len(student_outputs)
        else:
            if self.loss_type == 'cosine':
                target = torch.ones(student_outputs.size(0)).to(student_outputs.device)
                loss = self.loss_fn(student_outputs, teacher_outputs.detach(), target)
            else:
                loss = self.loss_fn(student_outputs, teacher_outputs.detach())
        return loss
