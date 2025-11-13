import numpy as np

from first_task.matrices import M1, M2
from multiprocessing import Pool, cpu_count
from datetime import datetime


def get_matrix_shape(matrix):
    return matrix.shape


def sum_chunk(args):
    i, chunk_size, n, cols = args
    end = min(i + chunk_size, n)
    M1_chunk = M1[i:end, :]
    M2_chunk = M2[i:end, :]
    result = M1_chunk + M2_chunk
    if np.any(np.isinf(result)) or np.any(np.isnan(result)):
        print(f"Warning: Chunk at index {i} contains inf or NaN")
        return (i, np.zeros_like(result))
    return (i, result)


def main():
    start_time = datetime.now()
    shape1 = get_matrix_shape(M1)
    shape2 = get_matrix_shape(M2)
    if shape1 != shape2 or shape1 != (1024, 1024):
        print("Error: Matrices must be 1024 x 1024 and have the same dimensions")
        return
    if np.any(np.isinf(M1)) or np.any(np.isinf(M2)) or np.any(np.isnan(M1)) or np.any(np.isnan(M2)):
        print("Error: Input matrices contain inf or NaN")
        return
    n, cols = shape1
    num_cores = cpu_count()
    chunk_size = max(n // num_cores, 1)
    M3 = np.zeros((n, cols), dtype=np.float64)
    chunk_indices = [(i, chunk_size, n, cols) for i in range(0, n, chunk_size)]
    with Pool(processes=num_cores) as pool:
        results = pool.map(sum_chunk, chunk_indices)
    for i, chunk in results:
        M3[i:i+len(chunk)] = chunk
    np.savetxt('M3.dat', M3, fmt='%.18e', delimiter=' ')
    print(f"End of operation, watch file - M3.dat")
    print(
        f"Optimized parallel matrix sum completed successfully using {num_cores} cores")
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")


if __name__ == "__main__":
    main()
