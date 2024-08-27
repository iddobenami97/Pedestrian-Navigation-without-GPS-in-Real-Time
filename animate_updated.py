import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.lines import Line2D

#fig = plt.figure()
#fig2 = plt.figure()
#ax = plt.axes(projection='3d')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#fig2 = plt.figure()

namafile = 'data.csv'
header1 = "x_value"
header2 = "y_value"
header3 = "z_value"
header4 = "step_type"

index = count()

# Define colors for step types
step_colors = {
    'step_forward': 'green',
    'stair_up': 'red',
    'left_crab': 'purple',
    'step_backward': 'gray',
    'stair_down': 'red',
    'right_crab': 'purple',
}

# Create legend elements
legend_elements = []
legend_elements.append(Line2D([0], [0], color='green', lw=4, label='step forward'))
legend_elements.append(Line2D([0], [0], color='gray', lw=4, label='step backward'))
legend_elements.append(Line2D([0], [0], color='red', lw=4, label='stair'))
legend_elements.append(Line2D([0], [0], color='purple', lw=4, label='crab'))

# Add the remaining elements for starting point and current location
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Starting Point', markerfacecolor='blue', markersize=10))
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Current Location', markerfacecolor='yellow', markersize=10))

def animate(i):
    data = pd.read_csv(namafile)
    x = data[header1]
    y = data[header2]
    z = data[header3]
    step_types = data[header4]

    # Find maximum and minimum absolute values for axis limits
    max_value = max(x.max(), y.max(), z.max(), 1)
    min_value = min(x.min(), y.min(), z.min(), -1)

    ax.clear()
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_zlim(min_value, max_value)
    ax.set(xlabel='X')
    ax.set(ylabel='Y')
    ax.set(zlabel='Z')
    text_offset = 0.3
    history_offset = 5
    # Plot arrows
    for j in range(1, len(x)):
        ax.quiver(
            x[j - 1],
            y[j - 1],
            z[j - 1],
            x[j] - x[j - 1],
            y[j] - y[j - 1],
            z[j] - z[j - 1],
            color=step_colors.get(step_types[j], 'black'),
            arrow_length_ratio=0.1
        )
        """
        jmod = (j-1)%57
        if (jmod <= 19):
            fig2.text(0,0.95-0.05*((jmod-1)%19), s=f"step #{j}: {step_types[j]}", fontsize=10, transform=fig.transFigure)
        elif (jmod <= 38):
            fig2.text(0.33,0.95-0.05*((jmod-1)%19), s=f"step #{j}: {step_types[j]}", fontsize=10, transform=fig.transFigure)
        elif (jmod <= 57):
            fig2.text(0.66,0.95-0.05*((jmod-1)%19), s=f"step #{j}: {step_types[j]}", fontsize=10, transform=fig.transFigure)
        """
            
    # Plot the origin
    ax.scatter(0, 0, 0, color='blue', s=100)

    last_x, last_y, last_z = x.iloc[-1], y.iloc[-1], z.iloc[-1]

    # Plot the current position
    ax.scatter(last_x, last_y, last_z, color='yellow', s=100)

    # Add title here
    fig.suptitle(f"3D Step Animation\ncurrent position: ({last_x}, {last_y}, {last_z})\nlast step: {step_types.iloc[-1]}")

    ax.text(last_x+text_offset,last_y+text_offset,last_z+text_offset, s=f"{last_x,last_y,last_z}", fontsize=10)

    # Add step type history as text annotations
    #for step_type in enumerate(step_types):
    #    ax.text(2,2,2, s=step_type, fontsize=8)


    # Add legend
    ax.legend(handles=legend_elements, loc='upper left')

ani = FuncAnimation(fig, animate, interval=250)

plt.show()
