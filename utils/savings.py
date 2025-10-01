import os
import matplotlib.pyplot as plt


def save_fig(fig, saving_path, figure_name, formats=['png']):
    # Set plot parameters.
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # Save
    for ext in formats:
        output_path = os.path.join(saving_path, f'{figure_name}.{ext}')
        fig.savefig(output_path, dpi=400)

    # Close fig
    plt.close('all')

