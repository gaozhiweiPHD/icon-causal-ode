import numpy as np
import torch
from causal_kv.data import EpisodeGenerator,batch_interface
from causal_kv.ode_solvers import local_solution


def test_shapes_and_hidden_parameters(episode):
    e=episode
    batch=batch_interface(e)
    for j in (1,2):
        assert batch[f'context_{j}_input'].shape==(3,4,64,2)
        assert batch[f'context_{j}_output'].shape==(3,4,64,1)
        assert batch[f'query_{j}_input'].shape==(3,64,2)
        assert batch[f'query_{j}_target'].shape==(3,64,1)
        p=e[f'context{j}_parameters']
        np.testing.assert_array_equal(p,p[:,:1].repeat(4,axis=1))
        assert len(np.unique(p[:,0],axis=0))==3
        c=e[f'context{j}']
        assert set(vars(c))=={'inputs','outputs'}
        u=c.inputs[...,1].numpy(); z=c.inputs[:,:,0,0].numpy()
        label=local_solution(p[:,:,0],p[:,:,1],z,u,e['time'])
        np.testing.assert_allclose(c.outputs[...,0].numpy(),label,atol=1e-7)


def test_reproducibility_and_separation(cfg):
    a=EpisodeGenerator(12,cfg).sample(3)
    b=EpisodeGenerator(12,cfg).sample(3)
    c=EpisodeGenerator(13,cfg).sample(3)
    torch.testing.assert_close(a['x_true'],b['x_true'],rtol=0,atol=0)
    np.testing.assert_array_equal(a['parameters'],b['parameters'])
    assert not np.array_equal(a['parameters'],c['parameters'])
    gen=EpisodeGenerator(12,cfg)
    assert not np.array_equal(gen.sample(3)['parameters'],gen.sample(3)['parameters'])


def test_ood_outside_training(cfg):
    p=EpisodeGenerator(12,cfg).sample(20,split='ood')['parameters']
    assert np.all((p[:,:2]<.6)|(p[:,:2]>1.4))
    assert np.all(abs(p[:,2:])>.25)
