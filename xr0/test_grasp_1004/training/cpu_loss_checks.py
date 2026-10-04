#!/usr/bin/env python3
"""CPU tests against the exact production loss, without constructing VLA/CUDA."""
import bootstrap
import ast
import copy
import json
from typing import Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F
from contact_loss import ContactLossMixin


def run():
    torch.set_num_threads(1)
    source=bootstrap.XR0/'mibot/models/VLA/XR0.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='XR0')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss'))
    ns=dict(torch=torch,F=F,Optional=Optional,Dict=Dict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),ns)
    class Base(nn.Module):
        compute_loss=ns['compute_loss']
        def __init__(self):
            super().__init__();self.head=nn.Parameter(torch.arange(960,dtype=torch.float32).view(1,30,32)/1000)
            self.async_train=False;self.training_repeat=1;self.freq_coefficient=0.;self.inputs=[]
        def forward(self,batch,return_loss=False):
            self.inputs.append({k:v.clone() for k,v in batch.items()})
            if 'raise_fixture' in batch:raise RuntimeError('fixture failure')
            action=batch.pop('action');mask=batch.pop('action_mask');state=batch.pop('state')
            if return_loss:return self.compute_loss(self.head,action,mask,torch.ones_like(action))
            return self.head.detach().clone()+state.sum()*0
    class Weighted(ContactLossMixin,Base):pass
    original=Base();weighted=Weighted()
    mask=torch.zeros(1,30,32,dtype=torch.int64);mask[:,:,:7]=1
    batch=dict(action=torch.linspace(-1,1,960).view(1,30,32),action_mask=mask,state=torch.zeros(1,1,32))
    before={k:v.clone() for k,v in batch.items()};unit=torch.ones(1,30,32)
    a=original(dict(batch),return_loss=True)['loss'];b=weighted(batch,return_loss=True,event_loss_weight=unit)['loss']
    a.backward();b.backward()
    assert torch.equal(a,b) and torch.equal(original.head.grad,weighted.head.grad)
    assert set(dict(original.named_parameters()))==set(dict(weighted.named_parameters()))
    assert set(original.state_dict())==set(weighted.state_dict())
    assert all(torch.equal(batch[k],v) for k,v in before.items())
    for row in weighted.inputs:assert set(row)==set(batch)
    weighted.zero_grad();w=unit.clone();w[:,4:14]=2
    loss=weighted(batch,return_loss=True,event_loss_weight=w)['loss'];loss.backward()
    active=mask.bool();expected=.5*((weighted.head-batch['action']).square()*w)[active].sum()/w[active].sum()
    expected_grad=torch.zeros_like(w);expected_grad[active]=((weighted.head.detach()-batch['action'])*w)[active]/w[active].sum()
    assert torch.allclose(loss,expected,atol=1e-8,rtol=1e-6)
    assert torch.allclose(weighted.head.grad,expected_grad,atol=1e-9,rtol=1e-6)
    assert torch.count_nonzero(weighted.head.grad[...,7:])==0
    assert weighted._contact_loss_context is None
    error_batch=dict(batch,raise_fixture=torch.tensor(1))
    try:weighted(error_batch,return_loss=True,event_loss_weight=w)
    except RuntimeError as e:assert str(e)=='fixture failure'
    else:raise AssertionError('Expected fixture exception')
    assert weighted._contact_loss_context is None
    weighted.eval();original.eval()
    assert torch.equal(weighted(batch),original(dict(batch)))
    rejected=0
    for kwargs in (dict(batch=batch,return_loss=False,event_loss_weight=w),dict(batch=dict(batch,event_loss_weight=w),return_loss=False)):
        try:weighted(**kwargs)
        except ValueError:rejected+=1
    weighted.train()
    try:weighted(batch,return_loss=True)
    except ValueError:rejected+=1
    else:raise AssertionError('Training weights must be required')
    badmask=dict(batch,action_mask=mask.float()*2)
    try:weighted(badmask,return_loss=True,event_loss_weight=w)
    except ValueError:rejected+=1
    assert rejected==4
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',original_loss_and_gradient_exact_for_unit_weight=True,
        event_1_2_matches_hand_calculated_loss_and_gradient=True,padding_gradients_zero=True,
        action_state_mask_input_cache_unchanged=True,event_labels_never_in_base_forward=True,
        named_parameters_and_state_dict_unchanged=True,exception_clears_context=True,
        inference_exact_and_rejects_event_weight=True,training_requires_explicit_weight=True,
        binary_mask_enforced=True,cuda_initialized=False)

if __name__=='__main__':print(json.dumps(run(),indent=2))
