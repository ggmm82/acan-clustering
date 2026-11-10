"""Causal teacher matching, separate from ACAN's unchanged attention equations."""
import torch
import torch.nn.functional as F


def objective(student_logits,teacher_logits,cross_entropy,weight,temperature):
    # At token position t both networks have seen only the same prefix <=t.
    student=student_logits[:,:-1].float()/temperature
    teacher=teacher_logits[:,:-1].detach().float()/temperature
    divergence=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction='none').sum(-1).mean()*temperature**2
    return (1-weight)*cross_entropy+weight*divergence,divergence
