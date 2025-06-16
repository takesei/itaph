import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


def convert_to_affine(
    A: Matrix | None = None,  # noqa: N803
    b: Vector | None = None,
) -> npt.NDArray[np.float64]:
    """
    Build an affine transformation matrix in homogeneous coordinates.

    Parameters
    ----------
    A : Optional[Matrix]
        Optional n x n linear transformation matrix. If None, uses identity.
    b : Optional[Matrix]
        Optional n-dimensional translation vector. If None, uses zero vector.

    Returns
    -------
    Matrix
        (n + 1) x (n + 1) affine transformation matrix:
            [[A, b],
             [0, 1]]
    """

    if A is None and b is None:
        raise ValueError('At least one of A or b must be specified')

    if A is None:
        b_arr = np.asarray(b, dtype=np.float64)
        n = b_arr.shape[0]
        A_mat = np.eye(n, dtype=np.float64)  # noqa: N806
    else:
        A_mat = np.asarray(A, dtype=np.float64)  # noqa: N806
        if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:  # noqa: PLR2004
            raise ValueError('A must be a square matrix')
        n = A_mat.shape[0]

    if b is None:
        b_vec = np.zeros(n, dtype=np.float64)
    else:
        b_arr = np.asarray(b, dtype=np.float64)
        if b_arr.ndim != 1 or b_arr.shape[0] != n:
            raise ValueError('b must be a 1D array of length n')
        b_vec = b_arr

    M = np.eye(n + 1, dtype=np.float64)  # noqa: N806
    M[:n, :n] = A_mat
    M[:n, n] = b_vec

    return M
