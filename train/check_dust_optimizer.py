"""Check compiled/eager updates and LR scheduling: uv run train/check_dust_optimizer.py --device cuda."""
import argparse
import copy
from pathlib import Path
import sys

import torch
from torch import nn
from torch._dynamo.utils import counters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.val.dust import make_optimizer, recipe, set_rate


def model(device):
    net = nn.Module()
    net.transformer = nn.ModuleDict({
        'h': nn.ModuleList([nn.ModuleDict({
            'square': nn.Linear(8, 8, bias=False),
            'tall': nn.Linear(8, 16, bias=False),
            'wide': nn.Linear(16, 8, bias=False),
            've_gate': nn.Linear(8, 2, bias=False),
        })]),
        'wte': nn.Embedding(16, 8),
    })
    net.lm_head = nn.Linear(8, 16, bias=False)
    net.resid_lambdas = nn.Parameter(torch.ones(2))
    return net.to(device).requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.manual_seed(0)
    for kind in ('adam', 'muon'):
        eager_model = model(args.device)
        compiled_model = copy.deepcopy(eager_model)
        eager = make_optimizer(eager_model, recipe(1), kind, compile_step=False)
        compiled = make_optimizer(compiled_model, recipe(1), kind)
        graph_count = None
        for step in range(1, 9):
            for opt in (eager, compiled):
                opt.zero_grad(set_to_none=True)
                set_rate(opt, step, 8, warmdown=1.0)
            for a, b in zip(eager_model.parameters(), compiled_model.parameters()):
                a.grad = torch.zeros_like(a) if step == 1 else torch.randn_like(a)
                b.grad = a.grad.clone()
            eager.step()
            compiled.step()
            for a, b in zip(eager_model.parameters(), compiled_model.parameters()):
                torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-4)
            if step == 3:
                graph_count = counters['stats']['unique_graphs']
            elif step > 3:
                assert counters['stats']['unique_graphs'] == graph_count, 'LR schedule recompiled the optimizer'
        for a_opt, b_opt in zip(getattr(eager, 'optimizers', [eager]), getattr(compiled, 'optimizers', [compiled])):
            for a_group, b_group in zip(a_opt.param_groups, b_opt.param_groups):
                for a, b in zip(a_group['params'], b_group['params']):
                    for key, value in a_opt.state[a].items():
                        torch.testing.assert_close(value, b_opt.state[b][key], rtol=2e-3, atol=2e-4)
        print(f'{kind}: compiled/eager parameters and state match; changing LR does not recompile', flush=True)


if __name__ == '__main__':
    main()
