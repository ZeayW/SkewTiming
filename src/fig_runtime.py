#
#
# import matplotlib.pyplot as plt
# import numpy as np
# from adjustText import adjust_text
#
# design_info = {
#     'ac97_ctrl': [7.1,14.39,0.73],
#     'gpio': [1.8,24.94,0.53],
#     'oc_wb_dma': [32.5,208.31,3.53],
#     'picorv32': [28.6,151.79,2.97],
#     'pid_controller': [9,26.42,1.29],
#     'systemcaes': [15.4,100.05,1.82],
#     'uart16550': [11.3,42.75,1.38],
#     'wb_lcd': [10.1,31.50,1.05],
#     'wb_dma': [32.5,36.77,1.77],
#     'y_huff': [41.6,305.64,3.15],
#     'oc_mem_ctrl': [34.9,218.19,4.08],
#     'wb_conmax': [59.8,230.33,6.40],
#     'y_quantizer': [40.7,353.55,3.83],
#     'pwm': [3.6,7.26,1.49],
#     'ps2': [3.7,6.20,1.20],
#     'ecg': [177.7,2286.24,25.20],
#     'y_dct': [227.6,3646.23,34.41],
#     'aes128':[256.5,7654.56,24.82]
# }
#
# values = list(design_info.values())
# # Example data
# design_size = np.array([v[0] for v in values])                # Design sizes
# tool_runtime = np.array([v[1] for v in values])  # tool DC Runtime (x-axis)
# our_runtime = np.array([v[2] for v in values])            # our Runtime (y-axis)
# speedup = tool_runtime / our_runtime          # Speedup values (color)
# print(np.min(speedup),np.max(speedup),np.mean(speedup))
# labels = list(design_info.keys())
#
#
# # Bubble sizes scaled for visualization
# bubble_sizes = design_size * 50
#
# # Create scatter plot
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(
#     tool_runtime,
#     our_runtime,
#     s=bubble_sizes,
#     c=speedup,
#     cmap='viridis',
#     alpha=0.7,
#     edgecolors='k'
# )
#
# # Add colorbar
# cbar = plt.colorbar(scatter)
# cbar.set_label('NUT-Timer Speedup',fontsize=14)
# cbar.ax.tick_params(labelsize=12)
#
# # Annotate each point with its label
# texts = []
# for i, label in enumerate(labels):
#     texts.append(
#         plt.annotate(label ,
#                      (tool_runtime[i], our_runtime[i]),
#                      fontsize=10)
#     )
#
#
# plt.tick_params(axis='x', labelsize=12)  # Increase x-axis numbers font size
# plt.tick_params(axis='y', labelsize=12)  # Increase y-axis numbers font size
#
# plt.xlim(5, 20000)   # Adjust x-axis limits to fit all circles
# plt.ylim(0.4, 100)
# # Labels and title
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Commercial Logic Synthesis Tool Runtime (s)',fontsize=14)
# plt.ylabel('NUT-Timer Runtime (s)',fontsize=14)
# # plt.title('Runtime Comparison with Design Size and Speedup')
#
#
# # Show plot
# # plt.tight_layout()
# plt.show()
# plt.savefig('runtime4.png')
#
#


import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import matplotlib.colors as mcolors

design_info = {
    'ac97_ctrl': [7.1, 14.39, 0.73],
    'gpio': [1.8, 24.94, 0.53],
    'oc_wb_dma': [32.5, 208.31, 3.53],
    'picorv32': [28.6, 151.79, 2.97],
    'pid_controller': [9, 26.42, 1.29],
    'systemcaes': [15.4, 100.05, 1.82],
    'uart16550': [11.3, 42.75, 1.38],
    'wb_lcd': [10.1, 31.50, 1.05],
    'wb_dma': [32.5, 36.77, 1.77],
    'y_huff': [41.6, 305.64, 3.15],
    'oc_mem_ctrl': [34.9, 218.19, 4.08],
    'wb_conmax': [59.8, 230.33, 6.40],
    'y_quantizer': [40.7, 353.55, 3.83],
    'pwm': [3.6, 7.26, 1.49],
    'ps2': [3.7, 6.20, 1.20],
    'ecg': [177.7, 2286.24, 25.20],
    'y_dct': [227.6, 3646.23, 34.41],
    'aes128': [256.5, 7654.56, 24.82]
}

values = list(design_info.values())
design_size = np.array([v[0] for v in values])
tool_runtime = np.array([v[1] for v in values])
our_runtime = np.array([v[2] for v in values])
speedup = tool_runtime / our_runtime
labels = list(design_info.keys())

# --- Improved Bubble Size Scaling ---
bubble_sizes = design_size * 75  # Reduced scaling factor
# Cap bubble sizes to avoid extreme overlap
# bubble_sizes = np.clip(bubble_sizes)

# --- Colormap and Normalization ---
# Use a more visually distinct colormap
cmap = plt.get_cmap('viridis')
# Normalize speedup values for better color distribution
norm = mcolors.LogNorm(vmin=np.min(speedup), vmax=np.max(speedup))

# Create scatter plot
plt.figure(figsize=(9, 6))  # Adjust figure size for better label visibility
scatter = plt.scatter(
    tool_runtime,
    our_runtime,
    s=bubble_sizes,
    c=speedup,
    cmap=cmap,
    norm=norm,
    alpha=0.7,
    edgecolors='k'
)

# Add colorbar
cbar = plt.colorbar(scatter,
                    ticks=[1, 10, 50, 100, 200, 300])  # set ticks
cbar.set_label('NUT-Timer Speedup', fontsize=18)
cbar.ax.tick_params(labelsize=15)

# Annotate each point with its label
texts = []
for i, label in enumerate(labels):
    if label in ['gpio','ps2','pwm','ac97_ctrl','wb_lcd','wb_dma','picorv32']:
        plt.annotate(label,
                     (tool_runtime[i], our_runtime[i]),
                     fontsize=12,
                     ha='center',
                     va='center')
    else:
        texts.append(
            plt.annotate(label,
                         (tool_runtime[i], our_runtime[i]),
                         fontsize=12,
                         ha='center',
                         va='center')
        )  # smaller font size, centered alignment

# --- Adjust Text Placement to Avoid Overlap ---
adjust_text(texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
            autoalign='xy',
            expand_points=(2, 2),
            )

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Commercial Logic Synthesis Tool Runtime (s)', fontsize=17)
plt.ylabel('NUT-Timer Runtime (s)', fontsize=17)
# plt.title('Runtime Comparison with Design Size and Speedup', fontsize=16)

# plt.grid(True, which="both", ls="-", alpha=0.3)  # add grid

plt.tight_layout()  # Ensure labels fit within the figure
plt.savefig('runtime.png')
plt.show()
