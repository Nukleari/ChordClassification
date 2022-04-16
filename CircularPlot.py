import matplotlib.pyplot as plt
import numpy as np
import io

def circular_plot(df, center_label='', top_label=''):
    # initialize the figure
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Set the coordinates limits
    lowerLimit = .1e7

    threshold = .3e7

    # Compute max and min in the dataset
    upper_limit = 1.0e7

    height_scalar = .5
    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    slope = (upper_limit - lowerLimit) / upper_limit
    heights = slope * df * height_scalar

    capped_heights = []
    for height in heights:
        capped_heights.append(min(height, upper_limit * height_scalar))
    heights = capped_heights

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(-3, len(df.index)+-3))
    angles = [-element * width for element in indexes]

    ax.bar(
        x=angles, 
        height=upper_limit * height_scalar, 
        width=width, 
        bottom=lowerLimit,
        linewidth= 2, 
        edgecolor="white",
        color="#f9f9f9",
    )

    ax.bar(
        x=angles, 
        height=threshold * height_scalar, 
        width=width, 
        bottom=lowerLimit,
        linewidth= 0, 
        edgecolor="white",
        color="#eeeeee",
    )

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth= 2, 
        edgecolor="white",
        color="#61a4b2",
    )

    colors = ["#f9f9f9", "#666666"]
    ax.bar(
        x=angles, 
        height=lowerLimit*.8, 
        width=width, 
        bottom=lowerLimit,
        linewidth= 2, 
        edgecolor="white",
        color=colors[0],
    )

    angles_black = []
    for i in range(1, len(df.index)):
        if len(df.index[i])>1:
            angles_black.append(angles[i])

    ax.bar(
        x=angles_black, 
        height=lowerLimit*.8, 
        width=width, 
        bottom=lowerLimit,
        linewidth= 2, 
        edgecolor="white",
        color=colors[1],
    )

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, df.keys()):
        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + lowerLimit*.4, 
            s=label, 
            ha='center', 
            va='center', 
            rotation=0, 
            rotation_mode="anchor") 

    ax.text(
            x=0, 
            y=0, 
            s=center_label, 
            ha='center', 
            va='center', 
            rotation=0, 
            rotation_mode="anchor", size=30) 

    ax.text(
        x=np.pi/2, 
        y=upper_limit*height_scalar, 
        s=top_label, 
        ha='center', 
        va='center', 
        rotation=0, 
        rotation_mode="anchor", size=10) 
    
    return fig

def circular_plot_numpy(df, center_label='', top_label=''):
    fig = circular_plot(df, center_label, top_label)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    data = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frame = data.reshape((int(h), int(w), -1))
    io_buf.close()
    plt.close('all')
    return frame