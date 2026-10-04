"""Loss-only contact supervision, independent of observation/action inputs.

The mixin deliberately owns no Parameter or Buffer. Inference delegates to the
unmodified XR0 path. The event tensor is supplied as a Python call keyword, never
as an item in the VLM batch, action, state, action_mask or noisy action.
"""
import torch


class ContactLossMixin:
    def forward(self, batch, return_loss=False, *, event_loss_weight=None):
        if any(key in batch for key in ('event_loss_weight', 'contact_flags', 'event_time_s')):
            raise ValueError('Supervision metadata must never be supplied in the model input batch')
        if getattr(self, '_contact_loss_context', None) is not None:
            raise RuntimeError('Reentrant weighted forward is unsupported')
        if event_loss_weight is not None:
            if not self.training or not return_loss:
                raise ValueError('Event weights are training-loss-only; inference must omit them')
            if self.async_train or self.training_repeat != 1 or self.freq_coefficient != 0:
                raise ValueError('L requires async=False, repeat=1, frequency=0')
            action=batch['action'];mask=batch['action_mask']
            if tuple(action.shape) != (1,30,32) or event_loss_weight.shape != action.shape:
                raise ValueError('L requires batch1, full30, 32 padded dimensions')
            if event_loss_weight.dtype != torch.float32 or event_loss_weight.device != action.device:
                raise ValueError('Independent raw loss weight must be float32 on the action device')
            if event_loss_weight.requires_grad or not bool(torch.isfinite(event_loss_weight).all()):
                raise ValueError('Supervision weights must be finite constants')
            if not bool(((event_loss_weight == 1) | (event_loss_weight == 2)).all()):
                raise ValueError('Raw target weights must be exactly 1 or 2')
            if not torch.equal(event_loss_weight, event_loss_weight[..., :1].expand_as(event_loss_weight)):
                raise ValueError('Same event weight must apply to all seven active action dimensions')
            if mask.shape != action.shape or not bool((mask[..., :7] == 1).all()) or bool((mask[..., 7:] != 0).any()):
                raise ValueError('Original binary active7 action mask must remain unchanged')
        elif self.training and return_loss:
            raise ValueError('L training cannot silently fall back to unweighted loss')
        self._contact_loss_context=event_loss_weight
        try:
            # XR0 auto_cast/get_action_input mutate their local dict; preserve caches.
            return super().forward(dict(batch),return_loss=return_loss)
        finally:
            self._contact_loss_context=None

    def compute_loss(self,pred,target,action_mask,weight=None):
        event_weight=getattr(self,'_contact_loss_context',None)
        if event_weight is not None:
            if not self.training or event_weight.shape != pred.shape:
                raise ValueError('Event weights cannot be reused outside their full30 training call')
            if weight is not None and not bool((weight == 1).all()):
                raise ValueError('L expects original XR0 weight to be all ones')
            weight=event_weight if weight is None else weight.float()*event_weight
        # Original loss normalizes over active7*30, then clamps [0.5,5].
        # Raw1/2 normalization lies in [0.5,2], so the clamp changes nothing.
        return super().compute_loss(pred,target,action_mask,weight)
