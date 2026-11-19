from util import index_to_grid

# Solve a constant coefficient Poisson problem on a regular grid.
#     - u_{xx} - u_{yy} = 1 \quad\textsf{in}\quad [0,1]^2\\
#     u = 0 \quad\textsf{on the boundary.}

def construct_poisson_operator(A,n,h):
    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = index_to_grid(row,n)
        A[row, row] = 4.0 / h**2
        if i > 0:
            column = row - n
            A[row, column] = -1.0 / h**2
        if i < n - 1:
            column = row + n
            A[row, column] = -1.0 / h**2
        if j > 0:
            column = row - 1
            A[row, column] = -1.0 / h**2
        if j < n - 1:
            column = row + 1
            A[row, column] = -1.0 / h**2

    A.assemblyBegin()
    A.assemblyEnd()
    return A