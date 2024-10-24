import numpy as np


# --------------------------------------------Differential operators-----------------------------------------------------
def pd_grad(f):
    """Compute the gradient of an image f

    This function return an array g containing the gradient of the input image f, i.e

    .. math:: g=\\nabla f = \\bigg( \\frac{df}{dx}, \\frac{df}{dy} \\bigg)

    where x and y correspond to rows and columns, respectively.

    The use formulation is the one provided by Chambolle to guarantee that the gradient is the adjoint of the divergence

    Parameters
    ----------
    f : 2D array
        input grayscale image

    Returns
    -------
    g - 3D array
        gradient vector field

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    n, m = f.shape
    g = np.zeros((n, m, 2))

    g[0:n - 2, :, 0] = f[1:n - 1, :] - f[0:n - 2, :]
    g[:, 0:m - 2, 1] = f[:, 1:m - 1] - f[:, 0:m - 2]

    return g


def pd_div(f):
    """Compute the divergence of the vector field f

    This function return an array g containing the divergence of the input vector field f.
    It corresponds to

    .. math:: g = div\qquad f = \\frac{df_0}{dx} + \\frac{df_1}{dy} \\qquad where \\qquad f=(f_0, f_1)

    and x and y correspond to rows and columns, respectively.

    The use formulation is the one provided by Chambolle to guarantee that the divergence is the adjoint of the gradient

    Parameters
    ----------
    f : 3D array
        input vector field

    Returns
    -------
    g - 2D array
        divergence

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    n, m, d = f.shape
    g = np.zeros((n, m))

    g[0, :] = f[0, :, 0]
    g[1:n - 2, :] = f[1:n - 2, :, 0] - f[0:n - 3, :, 0]
    g[n - 1, :] = f[n - 2, :, 0]

    g[:, 0] = g[:, 0] + f[:, 0, 1]
    g[:, 1:m - 2] = g[:, 1:m - 2] + f[:, 1:m - 2, 1] - f[:, 0:m - 3, 1]
    g[:, m - 1] = g[:, m - 1] - f[:, m - 2, 1]

    return g


# --------------------------------------------Proximal operators --------------------------------------------------------
def proxtv(p):
    """Compute the proximal operator associated with the L1 norm in the TV regularizer

    This function computes the proximal operator for

    .. math:: F(.)=\\| . \\|_1, i.e

    .. math:: g = prox_{\\sigma F^*}(p)_{i,j} = \\frac{p_{i,j}}{\\max(1,|p_{i,j}|)}

    Parameters
    ----------
    p : 3D array
        input vector field

    Returns
    -------
    g - 3D array
        vector field

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    pabs = np.maximum(1, np.sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2))
    dp = np.repeat(pabs[:, :, np.newaxis], 2, axis=2)

    return p / dp


def proxl1(x, g, lambd, tau):
    """Compute the proximal operator associated with the L1 norm for the data fidelity term

    This function computes the proximal operator for

    .. math:: G(.)=\\lambda \\|.-g \\|_1, i.e

    .. math:: p = prox_{\\tau G}(x) =  (x-\\lambda\\tau \\qquad  if \\qquad x-g>\\lamda \\tau) or (x+\\lambda \\tau \\qquad  if \\qquad x-g<-\\lambda \\tau) or (g \\qquad otherwise)

    Parameters
    ----------
    x : 2D array
        input image
    g : 2D array
        observed image
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter

    Returns
    -------
    p - 2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    p = g.copy()
    th = lambd * tau
    p[(x - g) > th] = x[(x - g) > th] - th
    p[(x - g) < -th] = x[(x - g) < -th] + th

    return p


def proxl2square(f, g, tau, lambd):
    """Compute the proximal operator associated with the square L2 norm

    This function computes the proximal operator for

    .. math:: G(.)=(\\lambda/2) \\|.-g \\|_2^2, i.e

    .. math:: prox_{\\tau G}(f) = (f+\\tau \\lambda g) / (1+\\tau \\lambda)

    Parameters
    ----------
    p : 2D array
        input image
    g : 2D array
        observed image
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    return (f + tau * lambd * g) / (1 + tau * lambd)


def proxtvl2inpainting(x, g, M, lambd, tau):
    """Compute the proximal operator associated with the square L2 norm with missing pizels for inpainting

    This function computes the proximal operator for

    .. math::  G(.)=(\\lambda/2) \\sum_{(i,j) in D-I}(._{i,j}-g_{i,j})^2,

    .. math:: prox_{\\tau G}(x)_{i,j} = (x_{i,j}+\\tau \\lambda M_{i,j} g_{i,j}) / (1+\\tau \\lambda M_{i,j})

    where M is a binary mask with 0 for missing pixels and 1 otherwise.

    Parameters
    ----------
    x : 2D array
        input image
    g : 2D array
        observed image
    M : 2D array
        binary mask of missing pixels
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    return (x + lambd * tau * M * g) / (1 + lambd * tau * M)


def proxtvl2multi(x, g, lambd, tau):
    """Compute the proximal operator associated with the square L2 norm with Ng observations

    This function computes the proximal operator for

    .. math::  G(.)=(\\lambda/2) \\sum_k \\|.-g_k \\|_2^2, i.e

    .. math:: prox_{\\tau G}(x) = (x+\\tau \\lambda \\sum_k g_k) / (1+\\tau \\lambda Ng)

    Parameters
    ----------
    x : 2D array
        input image
    g : 3D array
        multi-observed images
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    Ng = g.shape[2]

    return (x + tau * lambd * np.sum(g, axis=2)) / (1 + Ng * tau * lambd)


# ---------------------------------------------Main Algorithms-----------------------------------------------------------
def PD_ROF(g, lambd, tau, theta, N):
    """Apply Rudin-Osher-Fatemi, i.e TV-L2^2 scheme to the input image.

    This function uses the primal-dual method to solve

    .. math::  \\arg \\min_f \\qquad \\| \\nabla f \\|_1 + \\frac{\\lambda}{2} \\| f-g \\|_2^2

    Parameters
    ----------
    g : 2D array
        input image
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter
    theta : must be either 0 or 1
        primal-dual selection method
    N : integer
        number of iterations

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """
    sigma = 0.01 + 1 / (8 * tau)
    # initialisation
    f = g.copy()
    ft = g.copy()
    p = pd_grad(g)

    for i in range(N):
        p = proxtv(p + sigma * pd_grad(ft))
        fp = f.copy()
        f = proxl2square(f + tau * pd_div(p), g, tau, lambd)
        ft = f + theta * (f - fp)

    return f


def PD_TVL1(g, lambd, tau, theta, N):
    """Apply the TV-L1 scheme to the input image

    This function uses the primal-dual method to solve

    .. math:: \\arg \\min_f \\qquad \\| \\nabla f \\|_1 + \\frac{\\lambda}{2} \\| f-g \\|_1

    Parameters
    ----------
    g : 2D array
        input image
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter
    theta : must be either 0 or 1
        primal-dual selection method
    N : integer
        number of iterations

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    sigma = 0.01 + 1 / (8 * tau)

    # initialisation
    f = g.copy()
    ft = g.copy()
    p = pd_grad(g)

    for i in range(N):
        p = proxtv(p + sigma * pd_grad(ft))
        fp = f.copy()
        f = proxl1(f + tau * pd_div(p), g, tau, lambd)
        ft = f + theta * (f - fp)

    return f


def PD_TVL2inpainting(g, M, lambd, tau, theta, N):
    """Apply the TV-L2 inpainting scheme to the input image with mask of missing pixels M

    This function uses the primal-dual method to solve

    .. math:: \\arg \\min_f \\qquad \\|\\nabla f \\|_1 + \\frac{\\lambda}{2} \\sum_{(i,j) \\in known pixels}(f_{i,j}-g_{i,j})^2

    Parameters
    ----------
    g : 2D array
        input image
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter
    theta : must be either 0 or 1
        primal-dual selection method
    N : integer
        number of iterations

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    sigma = 0.01 + 1 / (8 * tau)

    # initialisation
    f = g.copy()
    ft = g.copy()
    p = pd_grad(g)

    # iteration
    for i in range(N):
        p = proxtv(p + sigma * pd_grad(ft))
        fp = f.copy()
        f = proxtvl2inpainting(f + tau * pd_div(p), g, M, tau, lambd)
        ft = f + theta * (f - fp)

    return f


def PD_TVL2multi(g, lambd, tau, theta, N):
    """Apply the TV-L2 scheme to form an image from multi degraded input images

    This function uses the primal-dual method to solve

    .. math:: \\arg \\min_f \\qquad \\| \\nabla f \\|_1 + \\frac{\\lambda}{2}\\sum_k \\| f-g_k \\|_2^2

    Parameters
    ----------
    g : 3D array
        input image (3rd dimension corresponds to observation number)
    tau : float
        dual regularization parameter
    lambd : float
        primal regularization parameter
    theta : must be either 0 or 1
        primal-dual selection method
    N : integer
        number of iterations

    Returns
    -------
    2D array
        output image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (08/06/2020)
    """

    sigma = 0.01 + 1 / (8 * tau)

    # initialisation
    f = g[:, :, 0].copy()
    ft = g[:, :, 0].copy()
    p = pd_grad(g[:, :, 0])

    # iteration
    for n in range(N):
        p = proxtv(p + sigma * pd_grad(ft))
        fp = f.copy()
        f = proxtvl2multi(f + tau * pd_div(p), g, tau, lambd)
        ft = f + theta * (f - fp)

    return f
