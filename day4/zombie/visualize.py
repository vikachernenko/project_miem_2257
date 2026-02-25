import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import pandas as pd

# 1. Читаем заголовок (размер сетки) из первой строки history.csv
with open('history.csv', 'r') as f:
    header = f.readline().strip()
    Nx, Ny = map(int, header.replace('# ', '').split(','))

# Читаем сами данные (пропуская первую строку с размерами)
data = pd.read_csv('history.csv', skiprows=1)
steps = len(data)

# 2. Читаем бинарные кадры
frames = np.fromfile('frames.bin', dtype=np.int8).reshape(steps, Ny, Nx)

# 3. Настраиваем окно графиков
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2)

ax_grid = fig.add_subplot(gs[0, 0])
ax_grid.set_title("Zombie Apocalypse Grid")
ax_grid.axis('off')
cmap = ListedColormap(['black', '#00bfff', '#ff3333'])
im = ax_grid.imshow(frames[0], cmap=cmap, vmin=0, vmax=2)

ax_plot = fig.add_subplot(gs[0, 1])
ax_plot.set_title("Population Dynamics")
ax_plot.set_xlim(0, steps)
ax_plot.set_ylim(0, max(data['Humans'].max(), data['Zombies'].max()) * 1.05)
ax_plot.set_xlabel("Time steps")
ax_plot.set_ylabel("Population")
ax_plot.grid(True, linestyle='--', alpha=0.6)

line_h, = ax_plot.plot([], [], color='#00bfff', lw=2, label="Humans")
line_z, = ax_plot.plot([], [], color='#ff3333', lw=2, label="Zombies")
ax_plot.legend(loc="upper right")

# 4. Функция анимации
def update(frame):
    im.set_array(frames[frame])
    line_h.set_data(data['Step'][:frame+1], data['Humans'][:frame+1])
    line_z.set_data(data['Step'][:frame+1], data['Zombies'][:frame+1])
    return im, line_h, line_z

ani = animation.FuncAnimation(fig, update, frames=steps, interval=40, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# СОХРАНЕНИЕ В GIF:
print("Сохраняю анимацию в zombie_apocalypse.gif... (это займет пару минут)")
ani.save('zombie_apocalypse.gif', writer='pillow', fps=15)
print("Готово! Файл zombie_apocalypse.gif сохранен в папке.")