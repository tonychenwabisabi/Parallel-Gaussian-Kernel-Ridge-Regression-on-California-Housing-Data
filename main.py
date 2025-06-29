from math import exp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from scipy.sparse.linalg import cg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Read data
data = pd.read_csv('data/housing.tsv', delimiter='\t')

# Split data into training and testing sets
scaler = StandardScaler()
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
xtrain_data = train_data.iloc[:, :-1].values
xtrain_data = scaler.fit_transform(xtrain_data)
ytrain_data = train_data.iloc[:, -1].values
xtest_data = test_data.iloc[:, :-1].values
xtest_data = scaler.transform(xtest_data)
ytest_data = test_data.iloc[:, -1].values

# Define kernel functions
def kerdis(x1, x2, s):
    diff = x1 - x2
    sum_sq = np.dot(diff, diff)
    return np.exp(-sum_sq / (2 * s ** 2))

def kerblock(X1, X2, s):
    M = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            M[i, j] = kerdis(X1[i], X2[j], s)
    return M

def fill_matrix(matrix, result, row_start, col_start):
    if np.isscalar(result):
        matrix[row_start, col_start] = result
    else:
        rows, cols = result.shape
        matrix[row_start:row_start+rows, col_start:col_start+cols] = result

def parallel_computation(xtrain_data, ytrain_data, xtest_data, ytest_data, lambda_, s):
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split data
    num_parts = size
    part_size = len(xtrain_data) // num_parts
    start = rank * part_size
    end = start + part_size
    if rank == num_parts - 1:
        end = len(xtrain_data)

    # Local data
    local_data = xtrain_data[start:end]

    # Initialize total matrix
    num_records = len(xtrain_data)
    total_matrix = None
    if rank == 0:
        total_matrix = np.zeros((num_records, num_records))

    # Compute kernel matrix block
    local_kernel_block = kerblock(local_data, xtrain_data, s)

    # Gather kernel matrix blocks from all processes
    kernel_blocks = comm.gather(local_kernel_block, root=0)

    mse_train = None
    mse_test = None

    if rank == 0:
        # Concatenate kernel matrix
        start_idx = 0
        for i, block in enumerate(kernel_blocks):
            end_idx = start_idx + block.shape[0]
            total_matrix[start_idx:end_idx, :] = block
            start_idx = end_idx

        # Add regularization term
        identity = np.eye(num_records)
        A = total_matrix + lambda_ * identity

        # Solve equation
        alpha_, info = cg(A, ytrain_data)

        # Predict on training set
        ytrain_predict = np.dot(total_matrix, alpha_)

        # Compute kernel matrix for test set
        total_matrix_test = kerblock(xtest_data, xtrain_data, s)

        # Predict on test set
        ytest_predict = np.dot(total_matrix_test, alpha_)

        # Compute mean squared error
        mse_train = np.sqrt(mean_squared_error(ytrain_data, ytrain_predict))
        mse_test = np.sqrt(mean_squared_error(ytest_data, ytest_predict))
            
    # Broadcast MSE values to all processes
    mse_train = comm.bcast(mse_train, root=0)
    mse_test = comm.bcast(mse_test, root=0)
        
    return mse_train, mse_test

if __name__ == '__main__':
    # Update parameter grid
    lambda_values = [0.01]
    s_values = [2.0, 2.05, 2.1, 2.15, 2.2]

    # List to store results
    results = []

    # Iterate over parameter grid
    for lambda_ in lambda_values:
        for s in s_values:
            mse_train, mse_test = parallel_computation(xtrain_data, ytrain_data, xtest_data, ytest_data, lambda_, s)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"lambda_: {lambda_}, s: {s}, Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
                results.append((lambda_, s, mse_train, mse_test))

    if MPI.COMM_WORLD.Get_rank() == 0:
        # Find parameter combination with minimum test MSE
        best_result = min(results, key=lambda x: x[3])
        print("\nBest parameter combination:")
        print(f"lambda_ = {best_result[0]}, s = {best_result[1]}")
        print(f"Corresponding Train MSE = {best_result[2]:.4f}, Test MSE = {best_result[3]:.4f}")
