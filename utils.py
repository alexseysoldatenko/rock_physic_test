import numpy as np
import itertools
from numba import njit


def calculate_Cklmn_from_k_mu(k_rock,mu_rock):
    """
    Calculate the stiffness matrix Cklmn from the rock properties k_rock and mu_rock.
    Parameters:
        k_rock (float): The bulk modulus of the rock.
        mu_rock (float): The shear modulus of the rock.
    Returns:
        Cklmn (ndarray): The 3D stiffness matrix Cklmn.
    """
    lambda_ = k_rock - (2 * mu_rock / 3)
    c11 = lambda_ + (2 * mu_rock)
    c12 = lambda_
    c44 = mu_rock

    C6x6 = np.zeros((6, 6))
    C6x6[0, 0] = c11
    C6x6[0, 1] = c12
    C6x6[0, 2] = c12
    C6x6[1, 0] = c12
    C6x6[1, 1] = c11
    C6x6[1, 2] = c12
    C6x6[2, 0] = c12
    C6x6[2, 1] = c12
    C6x6[2, 2] = c11
    C6x6[3, 3] = c44
    C6x6[4, 4] = c44
    C6x6[5, 5] = c44


    Cklmn = voigt_to_full_stiffness_matrix(C6x6)

    return Cklmn

def voigt_to_full_stiffness_matrix(voigt_matrix: np.ndarray) -> np.ndarray:
    """
    Converts the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.

    Parameters
    ----------
    voigt_matrix : np.ndarray
        6x6 stiffness matrix (Voigt notation).

    Returns
    -------
    np.ndarray
        3x3x3x3 stiffness matrix.
    """
    def full_to_voigt_index(i: int, j: int) -> int:
        if i == j:
            return i
        return 6 - i - j

    voigt_matrix = np.asarray(voigt_matrix)
    full_matrix = np.zeros((3, 3, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    voigt_i = full_to_voigt_index(i, j)
                    voigt_j = full_to_voigt_index(k, l)
                    full_matrix[i, j, k, l] = voigt_matrix[voigt_i, voigt_j]

    return full_matrix

def convert_stiffness_tensor(C, tol=1e-3, check_symmetry=True):
    """
    Convert a 3x3x3x3 stiffness tensor to a 6x6 Voigt notation tensor.

    Parameters:
        C (ndarray): The input 3x3x3x3 stiffness tensor.
        tol (float): The tolerance for checking symmetry. Default is 1e-3.
        check_symmetry (bool): Whether to check for symmetry. Default is True.

    Returns:
        ndarray: The 6x6 Voigt notation tensor.
    """
    voigt_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    C = np.asarray(C)
    voigt_tensor = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            k, l = voigt_indices[i]
            m, n = voigt_indices[j]
            voigt_tensor[i, j] = C[k, l, m, n]

    return voigt_tensor

def from_3x3x3x3_to_6x6(data ,dict_components):
    """
    Generates the 6x6 matrix result by mapping the values from the data dictionary using the components specified in dict_components.
    Parameters:
    - data (dict): A dictionary containing the values to be mapped to the result matrix. The keys of the dictionary represent the components in comp3x3x3x3 format.
    - dict_components (dict): A dictionary specifying the mapping between the components in comp6x6 format and the components in comp3x3x3x3 format.
    Returns:
    - result (numpy.ndarray): A 6x6 matrix containing the mapped values from the data dictionary. The mapping is performed according to the mapping specified in dict_components. The resulting matrix has symmetric elements and zero diagonal elements.
    """
    result = np.zeros((6, 6))

    for comp6x6, comp3x3x3x3 in dict_components.items():
        result[comp6x6] = data[comp3x3x3x3]

    result += result.T - np.diag(np.diag(result))
    return result

def calculate_Nmn(tetta = 0, phi = 0, a1 = 1000, a2 = 1000 ,a3 = 1, n = 0, m = 0):
    """
    Calculate the value of Nmn, which represents the product of two components of n_all vector.
    
    Parameters:
        tetta (float, optional): The angle tetta in radians. Defaults to 0.
        phi (float, optional): The angle phi in radians. Defaults to 0.
        a1 (float, optional): The value of a1. Defaults to 1000.
        a2 (float, optional): The value of a2. Defaults to 1000.
        a3 (float, optional): The value of a3. Defaults to 1.
        n (int, optional): The index of the first component of n_all. Defaults to 0.
        m (int, optional): The index of the second component of n_all. Defaults to 0.
    
    Returns:
        float: The product of the n_all[n] and n_all[m] components.
    """
    n_all = np.zeros(3)
    n_all[0] = np.sin(tetta) * np.cos(phi) / a1
    n_all[1] = np.sin(tetta) * np.sin(phi) / a2
    n_all[2] = np.cos(tetta) / a3
    
    return n_all[n] * n_all[m]


@njit
def LYAMBDAkl(Cklmn, k, l, n_all, *args):
    """
    Calculate the value of LYAMBDAkl.
    Parameters:
    - Cklmn: A 4-dimensional array containing coefficients (shape: (K,3,L,3)).
    - k: An integer representing the index in the first dimension of Cklmn.
    - l: An integer representing the index in the third dimension of Cklmn.
    - n_all: A 1-dimensional array containing values (shape: (3,)).
    - *args: Additional arguments.
    Returns:
    - result: The calculated result.
    """
    result = 0
    for m_i in range(3):
        for n_i in range(3):
            result += Cklmn[k,m_i,l,n_i] * n_all[n_i] * n_all[m_i]
    return result

# расчет Lymbdakl
@njit
def LYAMBDA(Cklmn, *args):
    """
    Calculates the LYAMBDA matrix for the given parameters.
    Parameters:
        Cklmn (numpy.ndarray): The Cklmn array used in the calculation.
        *args: Variable number of arguments (args[0], args[1], args[2], args[3], args[4]) representing angles and factors used in the calculations.
    Returns:
        numpy.ndarray: The inverse of the LYAMBDA matrix.
    """
    result = np.zeros((3,3))

    n_0 = np.sin(args[0]) * np.cos(args[1]) / args[2]
    n_1 = np.sin(args[0]) * np.sin(args[1]) / args[3]
    n_2 = np.cos(args[0]) / args[4]
    n_all =(n_0, n_1, n_2)
    for k_i in range(3):
        for l_i in range(3):
            result[k_i, l_i] = LYAMBDAkl(Cklmn, k_i, l_i, n_all, *args)
    if np.sum(result - result.T) != 0:
        print(result)

    return np.linalg.inv(result)

def get_axes(params):
    """ params: line_x это [(-np.pi/2, np.pi/, 0.001),(...)], где
        в каждом кортеже лежит начало конец и шаг по участку(граничные точки включаются 1 раз)
        return: возвращает сетку в которой нужно посчитать значения
        ____________________________________________________________
        записывает в поля grid_x1 и grid_x2 посчитанные точки для сетки
        """
    x = [np.array([params["limits_phi"][0][0]])]
    for part in params["limits_phi"]:
        data = np.arange(part[0] + part[2], part[1] + part[2], part[2])
        x.append(data)
    y = [np.array([params["limits_tetha"][0][0]])]
    for part in params["limits_tetha"]:
        data = np.arange(part[0] + part[2], part[1] + part[2], part[2])
        y.append(data)
    params['grid_phi'] = np.concatenate(x, axis=0)
    params['grid_tetha'] = np.concatenate(y, axis=0)

    return params


def get_array(params, func,  args = []):
    """
    Возвращает рассчитанные значения сетки
    """
    iter_number = 0
    params['res'] = np.zeros((params['grid_phi'].shape[0], params['grid_tetha'].shape[0]))
    for ix, phi in enumerate(params['grid_phi']):
        for iy, tetta in enumerate(params['grid_tetha']):
            iter_number += 1

            params['res'][ix, iy] = func(tetta, phi, *args)
    
    return params

