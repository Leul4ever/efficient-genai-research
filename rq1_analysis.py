import json, pathlib, itertools
from scipy.stats import spearmanr, kendalltau

lines = pathlib.Path('results/runs.jsonl').read_text().splitlines()
runs = {}
for l in lines:
    if l.strip():
        r = json.loads(l)
        runs[r['run_id']] = r

sweep = [r for r in runs.values() if r.get('study') == 'fast_lr_sweep' and r.get('status') == 'ok']

methods = ['random', 'perplexity', 'ifd']
lrs = [1e-6, 2e-5, 2e-4]

# build loss table: lr -> method -> loss
table = {}
for r in sweep:
    lr = r['config']['learning_rate']
    m  = r['config']['selection_method']
    table.setdefault(lr, {})[m] = r['metrics']['held_out_loss']

print("Loss table (lower = better):")
header = "  method        lr=1e-6    lr=2e-5    lr=2e-4"
print(header)
for m in methods:
    vals = [table[lr][m] for lr in lrs]
    print("  {:<12}  {:.4f}     {:.4f}     {:.4f}".format(m, vals[0], vals[1], vals[2]))

print()
print("Rankings per LR (1=best/lowest loss):")
rankings = {}
for lr in lrs:
    losses = [(m, table[lr][m]) for m in methods]
    ranked = sorted(losses, key=lambda x: x[1])
    rank_dict = {m: i+1 for i, (m, _) in enumerate(ranked)}
    rankings[lr] = rank_dict
    row = "  lr={}: ".format(lr) + "  ".join("{}={}".format(m, rank_dict[m]) for m in methods)
    print(row)

print()
print("Spearman correlations between LR conditions:")
for lr_a, lr_b in itertools.combinations(lrs, 2):
    vec_a = [rankings[lr_a][m] for m in methods]
    vec_b = [rankings[lr_b][m] for m in methods]
    rho, p = spearmanr(vec_a, vec_b)
    tau, p2 = kendalltau(vec_a, vec_b)
    print("  lr={} vs lr={}:  Spearman rho={:.3f}  Kendall tau={:.3f}".format(lr_a, lr_b, rho, tau))

print()
print("Method-level differences vs random per LR:")
for lr in lrs:
    base = table[lr]['random']
    print("  lr={}:".format(lr))
    for m in ['perplexity', 'ifd']:
        delta = table[lr][m] - base
        direction = "better" if delta < 0 else "worse"
        print("    {} vs random: {:+.4f}  ({})".format(m, delta, direction))

print()
print("IFD rank across LRs:")
for lr in lrs:
    print("  lr={}: IFD rank={}, loss={}".format(lr, rankings[lr]['ifd'], round(table[lr]['ifd'], 4)))

print()
print("Perplexity rank across LRs:")
for lr in lrs:
    print("  lr={}: perplexity rank={}, loss={}".format(lr, rankings[lr]['perplexity'], round(table[lr]['perplexity'], 4)))
