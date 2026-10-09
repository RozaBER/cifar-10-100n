"""Summarize repro/logs/*.log next to the CE numbers reported for CIFAR-N
(ResNet-34, 100 epochs; noisy-label columns as reprinted from the CIFAR-N leaderboard)."""
import glob
import os
import re

PAPER = {  # test accuracy (%) of CE in the paper setting
    ('cifar10', 'aggre'): 87.77, ('cifar10', 'rand1'): 85.02, ('cifar10', 'rand2'): 86.46,
    ('cifar10', 'rand3'): 85.16,  # one reprint lists 85.26
    ('cifar10', 'worst'): 77.69, ('cifar100', 'noisy100'): 55.50,
}
NOISE_RATE = {'clean': 0.0, 'aggre': 9.01, 'rand1': 17.23, 'rand2': 18.12, 'rand3': 17.64, 'worst': 40.21,
              'clean100': 0.0, 'noisy100': 40.20}

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(here, 'logs', '*.log'))):
    name = os.path.basename(path)[:-4]
    dataset, noise = name.split('_')[:2]
    text = open(path).read()
    accs = [float(x) for x in re.findall(r'test acc on test images is\s+([\d.]+)', text)]
    final = re.search(r'final: last test acc ([\d.]+), best test acc ([\d.]+)', text)
    rows.append((name, dataset, noise, accs, final))

print(f"{'run':40s} {'noise':>6s} {'epochs':>6s} {'last':>6s} {'best':>6s} {'paper CE':>8s}")
for name, dataset, noise, accs, final in rows:
    last = f'{accs[-1]:.2f}' if accs else '-'
    best = f'{max(accs):.2f}' if accs else '-'
    paper = PAPER.get((dataset, noise))
    paper = f'{paper:.2f}' if paper else '-'
    status = '' if final else '  (running)'
    print(f'{name:40s} {NOISE_RATE[noise]:5.2f}% {len(accs):6d} {last:>6s} {best:>6s} {paper:>8s}{status}')
    if accs:
        print('    per-epoch test acc: ' + ' '.join(f'{a:.1f}' for a in accs))
