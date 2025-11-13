import numpy as np

from multiprocessing import Pool
from generation import M1, M2, n
from datetime import datetime


def main():
    start = datetime.now()

    with open('/Users/viktoria/Desktop/научная инициатива/second_task/M3.dat', 'w') as file:
        M3 = np.dot(M1, M2)
        file.write(str(M3))

    end = datetime.now()
    execution_time = end - start
    print(f"Execution time = {execution_time}")


if __name__ == "__main__":
    main()
