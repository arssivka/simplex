import operator
import numpy as np


def simplex(a, b, z, blend=True):
    mat = np.array(a)
    m, n = mat.shape
    mat = np.append(mat, np.identity(m), axis=1)
    mat = np.append(mat, np.array(list([b_i] for b_i in b)), axis=1)
    z = np.append(-np.array(z), np.zeros(m))
    basis_variables = list(range(n, n + m))
    while any(z_i < 0 for z_i in z):
        p_col, _ = (min(enumerate(z), key=operator.itemgetter(1))
                    if blend else [(i, zi) for i, zi in enumerate(z) if zi < 0][0])
        p_row, _ = min(enumerate(mat.T[-1] / mat.T[p_col]),
                       key=operator.itemgetter(1))
        p_val = mat[p_row, p_col]
        if all(a_i <= 0 for a_i in mat.T[p_col]):
            return
        basis_variables[p_row] = p_col
        z -= z[p_col] / p_val * mat[p_row, 0:-1]
        mat = np.array(list(row - row[p_col] / p_val * mat[p_row]
                            if i != p_row else row / p_val for i, row in enumerate(mat)))
    x = np.zeros(n + m)
    x[basis_variables] = mat.T[-1]
    return x[0:n]


def main():
    # Example of using simplex method
    a = [[5, 2], [1, 1]]
    b = [10, 4]
    z = [5, 3]
    x = simplex(a, b, z, blend=True)
    print("x =", x, " z =", sum(map(operator.mul, x, z)) if x is not None else "Not found!")


if __name__ == '__main__':
    main()
