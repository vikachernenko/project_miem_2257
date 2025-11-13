import dask.array as da
from datetime import datetime


def random_matrix():
    m = int(input("Введите размер квадратной матрицы (m): "))
    M1 = da.random.uniform(low=1, high=100, size=(m, m),
                           chunks=(m//2, m//2))
    M2 = da.random.uniform(low=1, high=50, size=(m, m),
                           chunks=(m//2, m//2))
    return M1, M2


M1, M2 = random_matrix()
start = datetime.now()
M3 = M1.dot(M2).compute()
with open('/Users/viktoria/Desktop/научная инициатива/second_task/M3.dat', 'w') as file:
    file.write(str(M3))

end = datetime.now()
execution_time = end - start
print(f"Execution time = {execution_time}")
