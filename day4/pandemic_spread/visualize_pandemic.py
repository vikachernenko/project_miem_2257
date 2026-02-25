import pandas as pd
import matplotlib.pyplot as plt

# Читаем оба файла
reg_data = pd.read_csv('regular.csv')
sw_data = pd.read_csv('smallworld.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Epidemic Spread: Regular Lattice vs Small World Network', fontsize=16)

# График 1: Локальная сетка (Бета = 0)
ax1.set_title('Regular Lattice (No long-range contacts)')
ax1.plot(reg_data['Day'], reg_data['Susceptible'], color='blue', label='Susceptible', lw=2)
ax1.plot(reg_data['Day'], reg_data['Infected'], color='red', label='Infected', lw=2)
ax1.plot(reg_data['Day'], reg_data['Recovered'], color='green', label='Recovered', lw=2)
ax1.fill_between(reg_data['Day'], reg_data['Infected'], color='red', alpha=0.3)
ax1.set_xlabel('Days')
ax1.set_ylabel('Population')
ax1.set_ylim(0, 10000)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# График 2: Small World (Бета = 0.1)
ax2.set_title('Small World Network (Beta = 0.1)')
ax2.plot(sw_data['Day'], sw_data['Susceptible'], color='blue', label='Susceptible', lw=2)
ax2.plot(sw_data['Day'], sw_data['Infected'], color='red', label='Infected', lw=2)
ax2.plot(sw_data['Day'], sw_data['Recovered'], color='green', label='Recovered', lw=2)
ax2.fill_between(sw_data['Day'], sw_data['Infected'], color='red', alpha=0.3)
ax2.set_xlabel('Days')
ax2.set_ylim(0, 10000)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# Вычисляем статистику для отчета (Пик и длительность)
reg_peak = reg_data['Infected'].max()
sw_peak = sw_data['Infected'].max()
reg_dur = len(reg_data)
sw_dur = len(sw_data)

print(f"=== Comparative Analysis ===")
print(f"Regular Lattice Peak Infected: {reg_peak} people (Duration: {reg_dur} days)")
print(f"Small World Peak Infected:     {sw_peak} people (Duration: {sw_dur} days)")

plt.tight_layout()
plt.savefig('pandemic_comparison.png', dpi=300)
plt.show()