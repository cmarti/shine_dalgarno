import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logomaker
import torch

from plot_utils import FIG_WIDTH

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    params = torch.load('results/thermodynamic_model.pth')
    T = 273 + 37
    RT = 0.592 / 298 * T
    theta = (params['theta_raw'] - params['theta_raw'].mean(1).unsqueeze(1)).numpy()
    
    m = pd.DataFrame(RT * theta, columns=[x for x in 'ACGU'])
    theta0 = RT * params['theta0'].numpy().flatten()
    
    mut_A4U = m.loc[3, 'G'] - m.loc[3, 'A']
    mut_A4C = m.loc[3, 'C'] - m.loc[3, 'A']
    print('Mut effects\n\tA4G = {:.2} (kcal/mol)\n\tA4C = {:.2f}'.format(mut_A4U, mut_A4C))
    
    
    fig, subplots = plt.subplots(1, 2, figsize=(FIG_WIDTH * 0.575, FIG_WIDTH * 0.19))

    axes = subplots[0]
    positions = np.arange(-16, -16 + m.shape[0])
    axes.scatter(positions, -theta0, c='black', s=15)
    axes.plot(positions, -theta0, c='black')
    axes.set(xlabel='SD position relative to AUG',
            xticks=positions,
            ylabel='$-\Delta G_p$ (kcal/mol)',
            yticks=np.linspace(-7, -4, 7))

    axes = subplots[1]
    logomaker.Logo(-m, ax=axes)
    axes.set(ylabel=r'-$\Delta\Delta$G (kcal/mol)',
            # yticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
            ylim=(-2.5, 2.5),
            xlabel='Position in SD sequence',
            xticks=np.arange(0, m.shape[0]),
            xticklabels=np.arange(0, m.shape[0]) + 1)
    fig.tight_layout()
    fig.savefig('figures/thermodynamic_model.params.svg', dpi=300)
    fig.savefig('figures/thermodynamic_model.params.png', dpi=300)