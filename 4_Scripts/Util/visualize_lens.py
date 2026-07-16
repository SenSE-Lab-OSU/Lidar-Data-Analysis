import numpy as np
import matplotlib.pyplot as plt

# Initialize a stacked 2-panel plot comparing both focal lengths
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), dpi=150)

# FIX: Removed 'pad' and used 'y' to safely adjust vertical spacing of the main title
fig.suptitle('Optical Geometry & Magnification: 50mm vs. 200mm Lenses', fontsize=16, fontweight='bold', y=0.98)

# Professional schematic color palette
c_center = '#0072B2'   # Dark Blue for Central Ray
c_edge1 = '#D55E00'    # Deep Orange for Top Edge Ray
c_edge2 = '#CC79A7'    # Muted Purple for Bottom Edge Ray
c_lens = '#009E73'     # Emerald Green for Lens Elements
c_sensor = '#333333'   # Dark Charcoal for Sensor Plane

def plot_lens_system(ax, f, title):
    # Establish spatial coordinates (in mm)
    lens_x = 0
    sensor_x = f
    lens_radius = 25
    
    # Off-axis object emitting light from an angle
    obj_x = -150
    obj_y = 30
    
    # Draw reference Optical Axis
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Optical Axis')
    
    # 1. Calculate and plot Central Ray (Passes through node (0,0) unbent)
    m = (0 - obj_y) / (0 - obj_x)  # Slope calculation
    sensor_y = m * sensor_x         # Resulting intersection coordinate on sensor
    
    ax.plot([obj_x, 0], [obj_y, 0], color=c_center, linewidth=2, label='Central Ray (No Bend)')
    ax.plot([0, sensor_x], [0, sensor_y], color=c_center, linewidth=2)
    
    # 2. Calculate and plot Top Edge Ray (Converges at the same sensor point)
    ax.plot([obj_x, lens_x], [obj_y, lens_radius], color=c_edge1, linewidth=1.5, label='Top Edge Ray')
    ax.plot([lens_x, sensor_x], [lens_radius, sensor_y], color=c_edge1, linewidth=1.5)
    
    # 3. Calculate and plot Bottom Edge Ray (Converges at the same sensor point)
    ax.plot([obj_x, lens_x], [obj_y, -lens_radius], color=c_edge2, linewidth=1.5, label='Bottom Edge Ray')
    ax.plot([lens_x, sensor_x], [-lens_radius, sensor_y], color=c_edge2, linewidth=1.5)
    
    # Render physical lens graphic elements (Cylinder representation)
    lens_xs = np.linspace(-2, 2, 100)
    lens_ys = lens_radius * np.sqrt(1 - (lens_xs/6)**2)
    ax.fill_between(lens_xs, -lens_ys, lens_ys, color=c_lens, alpha=0.3, edgecolor=c_lens, linewidth=1.5, label='Lens Nodal Plane')
    
    # Render Image Sensor Plane
    sensor_height = 40
    ax.plot([sensor_x, sensor_x], [-sensor_height, sensor_height], color=c_sensor, linewidth=4, label='Sensor Plane')
    ax.text(sensor_x + 5, 0, f'Sensor Plane\n(at {f}mm)', verticalalignment='center', fontweight='bold', color=c_sensor, fontsize=9)
    
    # Highlight the focused image point calculation
    ax.scatter([sensor_x], [sensor_y], color='red', s=45, zorder=5, label='Focused Image Point')
    ax.text(sensor_x + 5, sensor_y, f'y = {sensor_y:.1f}mm', verticalalignment='center', color='red', fontsize=10, fontweight='bold')
    
    # Geometrical Calculations for text box annotations
    bend_top_in = np.degrees(np.arctan2(lens_radius - obj_y, lens_x - obj_x))
    bend_top_out = np.degrees(np.arctan2(sensor_y - lens_radius, sensor_x - lens_x))
    total_bend = abs(bend_top_in - bend_top_out)
    
    # Formatting adjustments
    ax.text(-145, -38, f'Peripheral Ray Bend: {total_bend:.1f}°\nImage Sensor Drift: {abs(sensor_y):.1f}mm', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', boxstyle='round,pad=0.5'), fontsize=9.5)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=10)
    ax.set_xlim(-170, 260)
    ax.set_ylim(-45, 45)
    ax.set_xlabel('Optical Axis Distance (mm)', fontsize=10)
    ax.set_ylabel('Height from Axis (mm)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

# Generate both plots
plot_lens_system(ax1, 50, 'A) 50mm Lens System (High Bending Power -> Low Magnification)')
plot_lens_system(ax2, 200, 'B) 200mm Lens System (Low Bending Power -> High Magnification)')

# Layout optimizations & centralized legend formatting
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5, 0.01), fontsize=9.5)

# Adjust layouts bounding box so titles do not overlap axes
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
