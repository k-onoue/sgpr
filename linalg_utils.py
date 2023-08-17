import torch
from torch import linalg
from misc_utils import safe_exp


def cholesky_decomposition(matrix, max_tries=5):
    diag_matrix = torch.diag(matrix)
    jitter = diag_matrix.mean() * 1e-6
    num_tries = 0
    
    try:
        L = linalg.cholesky(matrix, upper=False)
        return L
    except linalg.LinAlgError:
        num_tries += 1
        
    while num_tries <= max_tries and torch.isfinite(jitter):
        try:
            L = linalg.cholesky(matrix + torch.eye(matrix.shape[0]) * jitter, upper=False)
            return L
        except linalg.LinAlgError:
            jitter *= 10
            num_tries += 1
            
    raise linalg.LinAlgError("Matrix is not positive definite, even with jitter.")


def det(matrix):
    sign, logabsdet = linalg.slogdet(matrix)
    return sign * safe_exp(logabsdet)


def inv(matrix):
    # try:
    #     L = cholesky_decomposition(matrix)
    # except:
    #     matrix = nearest_spd(matrix)
    #     L = cholesky_decomposition(matrix)
    L = cholesky_decomposition(matrix)
    i_L = linalg.inv(L)
    return i_L.T @ i_L


def inv_diag(matrix):
    return torch.diag(1.0 / torch.diagonal(matrix))


def solve_linear_equation():
    pass


def is_symmetric(matrix):
    transposed_matrix = matrix.t()    
    return torch.allclose(matrix, transposed_matrix)


def nearest_spd(A):
    B = (A + A.T) / 2
    _, Sigma, V = linalg.svd(B)
    H = V @ torch.diag(Sigma) @ V.T
    
    Ahat = (B + H) / 2
    Ahat = (Ahat + Ahat.T) / 2
    
    k = 0
    while True:
        try:
            linalg.cholesky(Ahat)
            break
        except linalg.LinAlgError:
            k += 1
            min_eig = torch.min(torch.linalg.eigvals(Ahat).real)
            Ahat = Ahat + (-min_eig * k**2 + torch.finfo(float).eps * min_eig) * torch.eye(A.shape[0])
    
    return Ahat



# 実験用データ作成
def generate_pd_matrix(size=None):
    if size is None:
        size = 5
    A = torch.randn(size, 100)
    return A @ A.T

    
def generate_non_pd_matrix(size=None):    
    A = generate_pd_matrix(size=size)
    # Compute Eigdecomp
    vals, vectors = linalg.eig(A)
    # Set smallest eigenval to be negative with 5 rounds worth of jitter
    vals[vals.argmin()] = 0
    default_jitter = 1e-6 * torch.mean(vals)
    vals[vals.argmin()] = -default_jitter * (10 ** 3.5)
    A_corrupt = (vectors * vals).dot(vectors.T)
    return A_corrupt


def generate_diagonal_matrix(size=None):
    if size is None:
        size = 5
    return torch.diag(torch.random.rand(size))