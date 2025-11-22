import torch
import torch.nn as nn
from typing import Dict, List

class MultiTaskLoss(nn.Module):
    """
    Combined loss for training the full system.
    Implements L_vis, L_phys, L_consistency, L_slot, L_supervisor.
    """
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            'vis': 1.0,
            'consistency': 1.0,
            'slot': 1.0,
            'supervisor': 0.5
        }
        
    def l_vis(self, pred_bbox, target_bbox, pred_mask, target_mask):
        """Visual detection loss (simplified bbox regression + mask dice)."""
        if pred_bbox is None or target_bbox is None:
            return torch.tensor(0.0)
        bbox_loss = nn.functional.smooth_l1_loss(pred_bbox, target_bbox)
        
        if pred_mask is not None and target_mask is not None:
            # Dice loss for masks
            intersection = (pred_mask * target_mask).sum()
            dice = (2 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-8)
            mask_loss = 1 - dice
        else:
            mask_loss = torch.tensor(0.0)
            
        return bbox_loss + mask_loss
    
    def l_consistency(self, visual_state, physics_pred):
        """Consistency loss between visual observation and physics prediction."""
        if visual_state is None or physics_pred is None:
            return torch.tensor(0.0)
        # MSE between predicted state and observed state
        return nn.functional.mse_loss(physics_pred, visual_state)
    
    def l_slot(self, slots, reid_features):
        """Object permanence loss for tracking (contrastive)."""
        if slots is None or reid_features is None:
            return torch.tensor(0.0)
        # Simplified: encourage slots to match reid features
        return nn.functional.mse_loss(slots, reid_features)

    def l_sys_id(self, pred_params, target_params):
        """
        System Identification Loss.
        Train Visual Expert to predict physical parameters (mass, friction) from pixels.
        """
        if pred_params is None or target_params is None:
            return torch.tensor(0.0)
        return nn.functional.mse_loss(pred_params, target_params)
    
    def l_supervisor(self, routing_probs, rewards):
        """Policy gradient loss for supervisor routing decisions."""
        if routing_probs is None or rewards is None:
            return torch.tensor(0.0)
        # REINFORCE: -log(prob) * reward
        log_probs = torch.log(routing_probs + 1e-8)
        return -(log_probs * rewards).mean()
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Dict with keys 'bbox', 'mask', 'physics_pred', 'slots', 'routing_probs'
            targets: Dict with keys 'bbox_gt', 'mask_gt', 'visual_state', 'reid_features', 'rewards'
        """
        losses = {}
        
        losses['vis'] = self.l_vis(
            outputs.get('bbox'), targets.get('bbox_gt'),
            outputs.get('mask'), targets.get('mask_gt')
        )
        
        losses['consistency'] = self.l_consistency(
            targets.get('visual_state'),
            outputs.get('physics_pred')
        )
        
        losses['slot'] = self.l_slot(
            outputs.get('slots'),
            targets.get('reid_features')
        )
        
        losses['supervisor'] = self.l_supervisor(
            outputs.get('routing_probs'),
            targets.get('rewards')
        )
        
        # Weighted sum
        total_loss = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses
