from matplotlib import pyplot as plt
import numpy as np

def plot_flagged_frames(
        audio_data, vad_regions,
        hop_size=256, fig_size=(8,4),
        line_color='steelblue', line_style='solid', line_width=0.5,
        plot_grid=True, axes=None
    ):
    if axes is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        ax = axes    
    
    ax.plot(audio_data, color=line_color, ls=line_style, lw=line_width)
    if plot_grid:
        ax.grid()
    
    for reg in vad_regions:
        if reg.voiced:
            ax.axvspan(
                xmin=reg.start_idx * hop_size,
                xmax=reg.stop_idx * hop_size, color='grey', alpha=0.2
            )
    ax.set_xlim([0, len(audio_data)])

def plot_flagged_samples(
    audio_data, markers,
    hop_size=256, fig_size=(8,4),
    line_color='steelblue', line_style='solid', line_width=0.5,
    plot_grid=True, axes=None
    ):
    if axes is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        ax = axes    
    
    ax.plot(audio_data, color=line_color, ls=line_style, lw=line_width)
    if plot_grid:
        ax.grid()    
    
    ax.axvspan(xmin=markers[0][1], xmax=markers[1][1], color='grey', alpha=0.2)
    ax.set_xlim([0, len(audio_data)])

def waterfall_plot(
    frames, fig_size=(4,4), line_color='black', line_style='solid',
    line_width=0.5, overlap=0.5, axes=None, left_right=False,
    add_indices=False, font_size=6
    ):
    if axes is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        ax = axes

    data_max = np.max(frames)
    data_min = np.min(frames)
    plot_height = data_max - data_min
    plot_hop = (1 - overlap) * plot_height
    
    if left_right:
        frames = np.flipud(frames).T

    n = len(frames)
    full_height = (n - 1) * plot_hop + plot_height

    for i, frame in enumerate(frames):
        
        if left_right:
            plot_frame = plot_height + i * plot_hop - np.array(frame) + data_min
            ax.plot(plot_frame, range(len(plot_frame)), color=line_color, lw=line_width, ls=line_style)
            if add_indices:
                ax.text(plot_hop + i * plot_hop, -2, f'{i:>3}', rotation='vertical', fontsize=font_size)
        else:
            plot_frame = full_height - plot_height - i * plot_hop + np.array(frame) - data_min
            ax.plot(plot_frame, color=line_color, lw=line_width, ls=line_style)
            if add_indices:
                ax.text(-3, full_height - plot_height - i * plot_hop, f'{i:>3}', fontsize=font_size)

    if left_right:        
        ax.set_xlim([0, full_height])
    else:
        ax.set_ylim([0, full_height])
    ax.axis('off')
