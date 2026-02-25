import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import pandas as pd

# 1. Читаем размеры сетки
with open('fire_history.csv', 'r') as f:
    header = f.readline().strip()
    Nx, Ny = map(int, header.replace('# ', '').split(','))

# Читаем данные
data = pd.read_csv('fire_history.csv', skiprows=1)
steps = len(data)

# 2. Читаем бинарные кадры
frames = np.fromfile('fire_frames.bin', dtype=np.int8).reshape(steps, Ny, Nx)

# 3. Настраиваем окно графиков
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2)

# Панель 1: Сетка (Лес)
ax_grid = fig.add_subplot(gs[0, 0])
ax_grid.set_title("Forest Fire Cellular Automaton")
ax_grid.axis('off')

# Цвета: 0-Пусто(светло-серый), 1-Дерево(зеленый), 2-Огонь(красный), 3-Пепел(черный)
cmap = ListedColormap(['#ffffff', '#00e600', '#FF4500', '#000000'])
im = ax_grid.imshow(frames[0], cmap=cmap, vmin=0, vmax=3)

# Панель 2: Графики
ax_plot = fig.add_subplot(gs[0, 1])
ax_plot.set_title("Ecosystem Dynamics")
ax_plot.set_xlim(0, steps)
ax_plot.set_ylim(0, max(data['Trees'].max(), data['Burnt'].max()) * 1.1)
ax_plot.set_xlabel("Time steps")
ax_plot.set_ylabel("Number of cells")
ax_plot.grid(True, linestyle='--', alpha=0.6)

line_tree, = ax_plot.plot([], [], color='#00e600', lw=2, label="Healthy Trees")
line_fire, = ax_plot.plot([], [], color='#FF4500', lw=2, label="Burning")
line_burnt, = ax_plot.plot([], [], color='#000000', lw=2, label="Burnt/Ash")
ax_plot.legend(loc="center right")

# 4. Функция анимации
def update(frame):
    im.set_array(frames[frame])
    line_tree.set_data(data['Step'][:frame+1], data['Trees'][:frame+1])
    line_fire.set_data(data['Step'][:frame+1], data['Fires'][:frame+1])
    line_burnt.set_data(data['Step'][:frame+1], data['Burnt'][:frame+1])
    return im, line_tree, line_fire, line_burnt

ani = animation.FuncAnimation(fig, update, frames=steps, interval=40, blit=True, repeat=False)

plt.tight_layout()

# Хочешь посмотреть на экране?
# plt.show()

# Хочешь сразу сохранить GIF?
print("Сохраняю анимацию в forest_fire.gif...")
ani.save('forest_fire.gif', writer='pillow', fps=15)
print("Готово!")