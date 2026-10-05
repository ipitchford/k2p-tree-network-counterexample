"""Figure 1: the six reduced level-2 trinets (semi-directed). Reticulation arcs dashed with arrowheads."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
nets = json.load(open('../data/trinets_level2_unlabelled.json'))
POS = {
 0: dict(t0=(0,1), t1=(0,0), t2=(1,0), h0=(1,1), h1=(-0.75,0.5), **{'1':(1.6,1.55), '2':(-1.45,0.5), '3':(1.6,-0.55)}),
 1: dict(t0=(-1,0), t1=(1,0), h0=(0,1.1), t2=(0,0), h1=(0,-1.1), **{'1':(0,1.8), '2':(0,-1.8), '3':(0,0.55)}),
 2: dict(t0=(0,0), h0=(1,0), t1=(1,-1), h1=(0,-1), t2=(0.5,0.85), **{'1':(-0.6,-1.6), '2':(1.6,-1.6), '3':(0.5,1.55)}),
 3: dict(t0=(0,0), h0=(1,0), t2=(1,-1), t1=(0,-1), h1=(0.5,0.85), **{'1':(0.5,1.55), '2':(-0.6,-1.6), '3':(1.6,-1.6)}),
 4: dict(t0=(-1,0), t2=(1,0), h0=(0,1), t1=(0,-1), h1=(0,0), **{'1':(1.7,0), '2':(0.45,0.0), '3':(-1.7,0)}),
 5: dict(t0=(0,0), h0=(1,0), h1=(1,-1), t1=(0,-1), t2=(0.5,0.85), **{'1':(1.6,-1.6), '2':(-0.6,-1.6), '3':(0.5,1.55)}),
}
TITLE = {0: 'L2-0: coincides on $\\Theta_0$', 1: 'L2-1 ($K_{2,3}$): coincides on $\\Theta_0$,\nembeddable, $\\alpha\\geq\\beta$',
         2: 'L2-2: coincides on $\\Theta_0$', 3: 'L2-3 (stacked): distinguishable', 4: 'L2-4 (stacked): distinguishable',
         5: 'L2-5 (stacked): distinguishable'}
fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
for k, ax in enumerate(axes.flat):
    sd = nets[k]['sd']; pos = POS[k]
    for i, (u, v, d) in enumerate(sd):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        if d:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, linestyle='--',
                                         color='#b03a2e', lw=1.6, shrinkA=6, shrinkB=7))
        else:
            ax.plot([x1, x2], [y1, y2], color='#222222', lw=1.6, zorder=1)
        if k == 1:
            ax.text((x1+x2)/2 + 0.07, (y1+y2)/2 + 0.07, f'e{i}', fontsize=8, color='#1f4e79')
    for v, (x, y) in pos.items():
        if v in '123':
            ax.plot(x, y, 'o', color='black', ms=7, zorder=3); ax.text(x + 0.1, y + 0.08, v, fontsize=11, weight='bold')
        else:
            is_ret = v.startswith('h')
            ax.plot(x, y, 'o', mfc='white', mec='#b03a2e' if is_ret else '#222222', mew=1.8, ms=9, zorder=3)
            ax.text(x - 0.28, y + 0.12, v, fontsize=8, color='#555555')
    ax.set_title(TITLE[k], fontsize=10.5)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.9, 2.0); ax.set_ylim(-2.1, 2.1)
fig.suptitle('The six reduced trinets with a strict level-2 three-blob (up to leaf relabelling)\n'
             'dashed red arcs = reticulation edges; red circles = reticulation vertices', fontsize=11)
fig.tight_layout()
fig.savefig('../figures/fig1_level2_trinets.png', dpi=200); fig.savefig('../figures/fig1_level2_trinets.pdf')
print('saved')
