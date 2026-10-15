"""Small meaningful causality/duplicate/history-isolation checks for the heads."""
import json
from pathlib import Path
import torch
from nonlinear_memory import NonlinearMemoryHead


def main():
    torch.set_num_threads(2);torch.manual_seed(21)
    matrix=torch.linalg.qr(torch.randn(1,32,8)).Q;prior=torch.randn(1,8);context=torch.randn(1,64)
    design=torch.randn(1,5,20,32);packet_grams=design.transpose(-1,-2)@design;packet_rhs=torch.randn(1,5,32)
    grams=packet_grams.cumsum(1);rhs=packet_rhs.cumsum(1);counts=torch.arange(1,6)[None];noise=torch.randn(1,8);scale=torch.ones(1,8)*2
    result={}
    for kind in ['transport','stateless','gru','current_only']:
        head=NonlinearMemoryHead(kind)
        # Exercise a nonzero learned correction rather than only its zero-init
        # analytic baseline, which could hide accidental state dependencies.
        final=head.output if kind=='gru' else head.net[-1]
        torch.nn.init.normal_(final.weight,std=.1)
        output=head(matrix,prior,context,grams,rhs,counts,noise,scale)[0]
        prefix=head(matrix,prior,context,grams[:,:3],rhs[:,:3],counts[:,:3],noise,scale)[0]
        duplicate=head(matrix,prior,context,torch.cat([grams,grams[:,-1:]],1),torch.cat([rhs,rhs[:,-1:]],1),torch.cat([counts,counts[:,-1:]],1),noise,scale)[0]
        output.square().mean().backward()
        checks={'prefix_maxabs':float((output[:,:3]-prefix).detach().abs().max()),'exact_duplicate_maxabs':float((duplicate[:,-1]-duplicate[:,-2]).detach().abs().max()),'finite_parameter_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters()),'parameters':sum(p.numel() for p in head.parameters())}
        if kind=='stateless':
            # A learned persistent state must not leak into this control: if the
            # full current likelihood is identical, differing old learned states
            # cannot change it. Its declared precision-gain feature does encode
            # the previous analytic ledger, so compare the whole last two rows.
            changed_g=grams.clone();changed_r=rhs.clone();changed_g[:,:3]*=2;changed_r[:,:3]*=3
            other=head(matrix,prior,context,changed_g,changed_r,counts,noise,scale)[0]
            checks['same_final_two_ledgers_different_neural_past_maxabs']=float((output[:,-1]-other[:,-1]).detach().abs().max())
            assert checks['same_final_two_ledgers_different_neural_past_maxabs']==0
        if kind=='current_only':
            changed_g=packet_grams.clone();changed_g[:,:4]*=2
            changed_r=packet_rhs.clone();changed_r[:,:4]*=3
            other=head(matrix,prior,context,changed_g.cumsum(1),changed_r.cumsum(1),counts,noise,scale)[0]
            checks['same_final_packet_different_history_maxabs']=float((output[:,-1]-other[:,-1]).detach().abs().max())
            assert checks['same_final_packet_different_history_maxabs']<1e-5
        assert checks['prefix_maxabs']==0 and checks['exact_duplicate_maxabs']==0 and checks['finite_parameter_gradients']
        result[kind]=checks
    path=Path('results/h3_core/nonlinear_protocol_checks.json');path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
