import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.savings import save_fig


def plot_average_across_days(data_path, filename, saving_path):
    df = pd.read_csv(os.path.join(data_path, filename), index_col=0)
    figure, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=400)
    outcomes = [('outcome_n', 'black'), ('outcome_a', 'blue')]
    if max(df['day']) >= 0:  # This means there's one whisker training day at least
        outcomes.append(('outcome_w', 'green'))

    for outcome, color in outcomes:
        sns.lineplot(
            x='day', y=outcome, data=df,
            color=color, marker='o',
            estimator=np.nanmean, errorbar=('ci', 95), n_boot=1000,
            ax=ax, zorder=2
        )

    # force ticks at every unique "day" value
    days = sorted(df['day'].unique())
    ax.set_xticks(days)
    ax.set_xticklabels(days)  # optional, usually redundant since ticks show the numbers

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Days')
    ax.set_ylabel('Lick probability')

    mouse_names = ", ".join(str(m) for m in df['mouse_id'].unique())
    ax.set_title(f"Average_across_days – Mice: {mouse_names}")

    sns.despine()
    figure.tight_layout()

    save_path = os.path.join(saving_path, 'average_across_days')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_fig(fig=figure, saving_path=save_path,
             figure_name='mice_average_across_days',
             formats=['png', 'pdf', 'svg'])


def plot_single_session(data_path, saving_path):
    df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)

    for session_id in df.session_id.unique():
        mouse_id = session_id[0:5]
        session_table = df.loc[df['session_id'] == session_id]

        session_table = session_table.loc[session_table.early_lick == 0]
        session_table = session_table.reset_index(drop=True)

        # Add  trial info
        session_table['trial'] = session_table.index

        # Add block info
        session_table = session_table.assign(block=session_table['trial'] // 20)  # 20 is block length may need adjustment

        # Compute performance average per block
        for outcome, new_col in zip(['outcome_w', 'outcome_a', 'outcome_n', 'correct_choice'],
                                    ['hr_w', 'hr_a', 'hr_n', 'correct']):
            session_table[new_col] = session_table.groupby(['block', 'opto_stim'], as_index=False)[outcome].transform(
                'mean')

        # Subsample at one value per block for performance plot
        d = session_table.loc[session_table.early_lick == 0][int(20 / 2)::20]

        # Plot
        raster_marker = 2
        marker_width = 0.5
        fig_width, fig_height = 6, 3
        catch_palette = ['grey', 'black']
        auditory_palette = ['cyan', 'blue']
        whisker_palette = ['bisque', 'orange']

        # Plot the performance by block average
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        for outcome, palette in [('hr_n', catch_palette), ('hr_a', auditory_palette), ('hr_w', whisker_palette)]:
            if outcome in d.columns and (not np.isnan(d[outcome].values[:]).all()):
                sns.lineplot(data=d, x='trial', y=outcome, color=palette[1], markeredgecolor=palette[1], ax=ax,
                             marker='o',
                             lw=2, legend=False)

        # Plot the single trials :
        for outcome, color_offset, palette in [('outcome_n', 0.1, catch_palette), ('outcome_a', 0.15, auditory_palette),
                                               ('outcome_w', 0.2, whisker_palette)]:
            if outcome in d.columns and (not np.isnan(d[outcome]).all()):
                for lick_flag, color_index in zip([0, 1], [0, 1]):
                    lick_subset = session_table.loc[session_table.lick_flag == lick_flag]
                    ax.scatter(x=lick_subset['trial'], y=lick_subset[outcome] - lick_flag - color_offset,
                               color=palette[color_index], marker=raster_marker, linewidths=marker_width)
        sns.despine()
        ax.set_ylim(-0.25, 1.05)
        ax.set_xlim(-1, len(session_table) + 0.05)
        ax.set_ylabel('Lick probability')
        ax.set_xlabel('Trial number')
        ax.set_title(f'{session_id}')
        fig.tight_layout()

        save_path = os.path.join(saving_path, 'single_mouse', f'{mouse_id}', f'{session_id}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_fig(fig=fig, saving_path=save_path, figure_name=f'{session_id}',
                 formats=['png', 'pdf', 'svg'])


def plot_single_mouse_across_days(data_path, saving_path):
    main_df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)
    n_mice = len(main_df.mouse_id.unique())
    print(f" ")
    print(f"Plot average across days for {n_mice} mice")
    for mouse in main_df.mouse_id.unique():
        print(f"Mouse : {mouse}")
        mouse_table = main_df.loc[main_df.mouse_id == mouse]
        mouse_table = mouse_table.reset_index(drop=True)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'whisker_psy'))]

        # Select columns for plot
        cols = ['outcome_a', 'outcome_w', 'outcome_n', 'day', 'opto_stim']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]

        # Compute hit rates. Use transform to propagate hit rate to all entries.
        df['hr_w'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_w'].transform('mean')
        df['hr_a'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_a'].transform('mean')
        df['hr_n'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_n'].transform('mean')

        # Average by day for this mouse
        df_by_day = df.groupby(['day', 'opto_stim'], as_index=False).agg('mean')

        # Do the plot
        figsize = (6, 4)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_by_day, x='day', y='hr_n', color='k', ax=ax, marker='o')
        sns.lineplot(data=df_by_day, x='day', y='hr_a', color='royalblue', ax=ax, marker='o')
        if max(df_by_day['day'].values) >= 0:  # This means there's one whisker training day at least
            sns.lineplot(data=df_by_day, x='day', y='hr_w', color='orange', ax=ax, marker='o')

        # ✅ Force tick on each day
        days = sorted(df_by_day['day'].unique())
        ax.set_xticks(days)
        ax.set_xticklabels(days)  # or [d+1 for d in days] if you want 1-based labels

        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Days')
        ax.set_ylabel('Lick probability')
        ax.set_title(f"{mouse}")
        sns.despine()
        figure.tight_layout()

        save_path = os.path.join(saving_path, 'single_mouse', f'{mouse}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_fig(fig=figure, saving_path=save_path, figure_name=f'{mouse}',
                 formats=['png', 'pdf', 'svg'])



def plot_average_reaction_time_across_days(data_path, saving_path):
    main_df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)
    n_mice = len(main_df.mouse_id.unique())
    print("\nPlot average across days for {} mice".format(n_mice))

    # Prep data
    df = main_df[['mouse_id', 'session_id', 'day',
                  'lick_time', 'response_window_start_time', 'trial_type']].copy()
    df['computed_reaction_time'] = df['lick_time'] - df['response_window_start_time']
    df = df.loc[df.trial_type == 'whisker_trial']

    # Average per mouse × session × day
    df = (df.drop(columns='trial_type')
            .groupby(['mouse_id', 'session_id', 'day'], as_index=False)
            .agg({'computed_reaction_time': 'mean'}))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0), dpi=300)

    mouse_names = ", ".join(str(m) for m in df['mouse_id'].unique())
    ax.set_title(f"Average_reaction_time – Mice: {mouse_names}")

    sns.lineplot(
        data=df,
        x='day',
        y='computed_reaction_time',
        estimator=np.nanmean,
        errorbar=('ci', 95),   # shaded CI band
        n_boot=1000,
        marker='o',
        ax=ax
    )

    # Force ticks on every day
    days = sorted(df['day'].unique())
    ax.set_xticks(days)
    ax.set_xticklabels(days)

    ax.set_xlabel('Days')
    ax.set_ylabel('Reaction time (s)')
    sns.despine()
    fig.tight_layout()

    save_path = os.path.join(saving_path, 'average_across_days')
    os.makedirs(save_path, exist_ok=True)
    save_fig(fig=fig, saving_path=save_path, figure_name='average_reaction_time',
             formats=['png', 'pdf', 'svg'])

def plot_single_mouse_reaction_time_across_days(data_path, saving_path):
    main_df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)

    for mouse in main_df.mouse_id.unique():
        print(f"Mouse : {mouse}")
        mouse_table = main_df.loc[(main_df.mouse_id == mouse) & (main_df.trial_type == 'whisker_trial')]
        mouse_table = mouse_table.reset_index(drop=True)
        mouse_table['computed_reaction_time'] = mouse_table['lick_time'] - mouse_table['response_window_start_time']

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.pointplot(data=mouse_table, x='day', y='computed_reaction_time', ax=ax)
        ax.set_xlabel('Day')
        ax.set_ylabel('Reaction time (s)')
        ax.set_title(f"{mouse}")
        sns.despine()
        fig.tight_layout()

        save_path = os.path.join(saving_path, 'single_mouse', f'{mouse}')
        os.makedirs(save_path, exist_ok=True)
        save_fig(fig=fig, saving_path=save_path,
                 figure_name=f'{mouse}_reaction_time',
                 formats=['png', 'pdf', 'svg'])

def plot_single_mouse_reaction_time_days_0_1(data_path, saving_path):
    main_df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)

    # Compute reaction times
    main_df['computed_reaction_time'] = main_df['lick_time'] - main_df['response_window_start_time']

    # Keep only whisker trials, and days 0 and 1
    df = main_df.loc[(main_df.trial_type == 'whisker_trial') & (main_df.day.isin([0, 1]))]

    # Find global y-limits across all mice for comparability
    y_min = df['computed_reaction_time'].min()
    y_max = df['computed_reaction_time'].max()

    for mouse in df.mouse_id.unique():
        print(f"Mouse : {mouse}")
        mouse_table = df.loc[df.mouse_id == mouse].reset_index(drop=True)

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.pointplot(data=mouse_table, x='day', y='computed_reaction_time', ax=ax)

        # Force consistent y-axis across all mice
        ax.set_ylim([y_min, y_max])

        # Only ticks for day 0 and 1
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])

        ax.set_xlabel('Day')
        ax.set_ylabel('Reaction time (s)')
        ax.set_title(f"{mouse}")
        sns.despine()
        fig.tight_layout()

        save_path = os.path.join(saving_path, 'single_mouse', f'{mouse}')
        os.makedirs(save_path, exist_ok=True)
        save_fig(fig=fig, saving_path=save_path,
                 figure_name=f'{mouse}_reaction_time_days0_1',
                 formats=['png', 'pdf', 'svg'])


def plot_single_session_reaction_time(data_path, saving_path):
    main_df = pd.read_csv(os.path.join(data_path, 'concatenated_bhv_tables.csv'), index_col=0)

    for session in main_df.session_id.unique():
        print(f"Session : {session}")
        session_table = main_df.loc[(main_df.session_id == session) & (main_df.trial_type == 'whisker_trial')]
        session_table = session_table.reset_index(drop=True)
        session_table['computed_reaction_time'] = session_table['lick_time'] - session_table['response_window_start_time']

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 4), width_ratios=(2, 1))
        sns.scatterplot(data=session_table, x=session_table.index, y='computed_reaction_time', ax=ax0)
        sns.boxplot(data=session_table, y='computed_reaction_time', ax=ax1)
        ax0.set_xlabel('Whisker trial')
        ax0.set_ylabel('Reaction time (s)')
        ax0.set_title(f"{session}")
        sns.despine()
        fig.tight_layout()

        save_path = os.path.join(saving_path, 'single_mouse', f'{session[0:5]}', f'{session}')
        os.makedirs(save_path, exist_ok=True)
        save_fig(fig=fig, saving_path=save_path,
                 figure_name=f'{session}_reaction_time',
                 formats=['png', 'pdf', 'svg'])


def plot_particle_test(data_path, filename, saving_path):
    df = pd.read_csv(os.path.join(data_path, filename), index_col=0)
    figure, ax = plt.subplots(1, 1, figsize=(3, 4), dpi=400)

    outcomes = [('outcome_n', 'black'), ('outcome_a', 'blue')]
    if df['day'].max() >= 0:
        outcomes.append(('outcome_w', 'green'))

    # x ticks & custom labels
    days = sorted(df['day'].unique())
    ax.set_xticks(days)
    custom_labels = {0: 'Particle on', 1: 'Particle off', 2: 'Particle on'}
    ax.set_xticklabels([custom_labels.get(d, d) for d in days])

    # --- plot per-mouse connected lines (per-mouse/day means) ---
    for outcome, color in outcomes:
        per_mouse = (df.groupby(['mouse_id', 'day'], as_index=False)[outcome]
                       .mean()
                       .sort_values(['mouse_id', 'day']))
        for mid, sub in per_mouse.groupby('mouse_id'):
            ax.plot(sub['day'], sub[outcome],
                    marker='o', linewidth=1, markersize=3,
                    alpha=0.35, color=color, zorder=1)

    # --- plot group average with error bars ONLY (no connecting line) ---
    for outcome, color in outcomes:
        sns.pointplot(
            x='day', y=outcome, data=df,
            color=color, marker='o',
            estimator=np.nanmean,
            # If your seaborn >=0.12 use errorbar=('ci', 95); otherwise use ci=95.
            errorbar=('ci', 95),
            n_boot=1000,
            linestyle='none',              # <- no connecting line for the mean
            ax=ax, zorder=3
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Lick probability')
    ax.set_xlabel("")

    mouse_names = ", ".join(str(m) for m in df['mouse_id'].unique())
    ax.set_title(f"Particle test – Mice: {mouse_names}")

    sns.despine()
    figure.tight_layout()

    save_path = os.path.join(saving_path, 'particle_test')
    os.makedirs(save_path, exist_ok=True)
    save_fig(fig=figure, saving_path=save_path,
             figure_name='particle_test',
             formats=['png', 'pdf', 'svg'])
