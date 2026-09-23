"""Read-only local gradients; no optimizer or ranking."""
import torch
from td_distill_model import bpr_loss, directional_distillation_loss

COEFFICIENTS = {'image': 0.3/1.3, 'text': 0.3*0.3/1.3}

def relation(a, b):
    a=a.double(); b=b.double()
    na=float(a.norm()); nb=float(b.norm())
    return {'norm':na,'relative_to_bpr':na/nb if nb else None,
            'cosine_to_bpr':float(torch.dot(a,b)/(na*nb)) if na and nb else None}

def inspect_batch(model, targets, batch):
    users,pos,neg=batch
    u=model.user_id_embedding(users); p=model.item_id_embedding(pos); n=model.item_id_embedding(neg)
    raw={'bpr':bpr_loss(u,p,n),
         'image':directional_distillation_loss(model.project_items(p)['image'],targets['image_items'][pos]),
         'text':directional_distillation_loss(model.project_items(p)['text'],targets['text_items'][pos])}
    losses=dict(raw)
    losses['weighted_image']=COEFFICIENTS['image']*raw['image']
    losses['weighted_text']=COEFFICIENTS['text']*raw['text']
    losses['weighted_distill']=losses['weighted_image']+losses['weighted_text']
    losses['total']=raw['bpr']+losses['weighted_distill']
    params=(model.user_id_embedding.weight,model.item_id_embedding.weight)
    gradients={}
    for name,loss in losses.items():
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        gradients[name]=tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gs))
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients[name]):
            raise RuntimeError('Nonfinite loss/gradient: '+name)
    for i in range(2):
        torch.testing.assert_close(gradients['total'][i],gradients['bpr'][i]+gradients['weighted_distill'][i],rtol=2e-5,atol=1e-8)
    result={'losses':{k:float(v.detach()) for k,v in losses.items()},'gradients':{}}
    for side,index in [('users',0),('items',1),('all',None)]:
        vectors={k:(torch.cat([v[0].flatten(),v[1].flatten()]) if index is None else v[index].flatten()) for k,v in gradients.items()}
        result['gradients'][side]={k:relation(v,vectors['bpr']) for k,v in vectors.items()}
        result['gradients'][side]['image_text_cosine']=relation(vectors['weighted_image'],vectors['weighted_text'])['cosine_to_bpr']
    return result
