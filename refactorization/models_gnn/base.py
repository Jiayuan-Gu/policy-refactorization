import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGNN(nn.Module):
    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_losses(self, pd_dict: dict, data_batch: dict, **kwargs):
        loss_dict = dict()
        ce_loss = F.cross_entropy(pd_dict['logits'], data_batch['action'])
        loss_dict['ce_loss'] = ce_loss
        return loss_dict

    @torch.no_grad()
    def update_metrics(self, pd_dict: dict, data_batch: dict, metrics: dict, **kwargs):
        logits = pd_dict['logits']
        pred_action = logits.argmax(-1)
        target_action = data_batch['action']
        if 'acc' not in metrics:
            from common.utils.metric_logger import Accuracy
            metrics['acc'] = Accuracy()
        metrics['acc'].update(pred_action, target_action)

        for action_idx in range(self.output_dim):
            name = f'acc_{action_idx}'
            if name not in metrics:
                from common.utils.metric_logger import Accuracy
                metrics[name] = Accuracy()
            mask = target_action == action_idx
            metrics[name].update(pred_action[mask], target_action[mask])
