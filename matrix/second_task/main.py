import numpy as np

from generation import M1, M2, n
from datetime import datetime


start = datetime.now()


with open('/Users/viktoria/Desktop/научная инициатива/second_task/M3.dat', 'w') as file:
    M3 = M1@M2
    file.write(str(M3))

end = datetime.now()
execution_time = end - start

print(f"Execution time = {execution_time}")
