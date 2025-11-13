import numpy as np
import random
from datetime import datetime


def generate_tridiagonal_matrix(n):
    main_diag = [random.uniform(10, 100) for _ in range(n)]

    super_diag = [random.uniform(1, 10) for _ in range(n-1)]
    sub_diag = [random.uniform(1, 10) for _ in range(n-1)]

    return sub_diag, main_diag, super_diag


def save_matrix_to_file(filename, n, sub, main, super):
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        f.write(" ".join(map(str, sub)) + "\n")
        f.write(" ".join(map(str, main)) + "\n")
        f.write(" ".join(map(str, super)) + "\n")


def read_tridiagonal_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        sub = list(map(float, f.readline().split()))
        main = list(map(float, f.readline().split()))
        super = list(map(float, f.readline().split()))
    return n, sub, main, super


def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom

    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / \
        (b[n-1] - a[n-2] * c_prime[n-2])

    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


def invert_tridiagonal(n, a, b, c):
    inv = np.zeros((n, n))

    for i in range(n):
        d = np.zeros(n)
        d[i] = 1.0
        inv[:, i] = thomas_algorithm(a, b, c, d)

    return inv


if __name__ == "__main__":
    n = int(input("введите размер матрицы: "))
    input_file = "M.dat"
    output_file = "T.dat"

    sub, main, super = generate_tridiagonal_matrix(n)

    save_matrix_to_file(input_file, n, sub, main, super)
    print(f"Сгенерирована и сохранена {n}x{n} матрица в {input_file}")

    start = datetime.now()
    n, a, b, c = read_tridiagonal_matrix(input_file)

    inverse = invert_tridiagonal(n, a, b, c)

    np.savetxt(output_file, inverse)

    end = datetime.now()
    execution_time = end - start
    print(f"Обратная матрица сохранена в {output_file}")
    print(
        f"время выполнения чтения и нахождения обратной матрицы :{execution_time}")
