from typing import Mapping, Sequence, Tuple
import numpy as np
import math as ma


def spatial_frequencies(gpts: Tuple[int, int], sampling: Tuple[float, float]):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    return tuple(
        np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling)
    )


def electron_wavelength_angstrom(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = (
        h
        / ma.sqrt(2 * m * e * E_eV)
        / ma.sqrt(1 + e * E_eV / 2 / m / c**2)
        * 10**10
    )
    return lam


def aberrations_basis_function(
    intesity_size,
    sampling_size,
    energy,
    max_angular_order,
    min_radial_order,
    max_radial_order,
    xp=np,
):
    """ """

    # Add constant phase shift in basis
    #mn = [[-1, 0, 0]]
    mn = []

    for m in range(min_radial_order, max_radial_order):
        n_max = np.minimum(max_angular_order, m + 1)
        for n in range(0, n_max + 1):
            if (m + n) % 2:
                mn.append([m, n, 0])
                if n > 0:
                    mn.append([m, n, 1])

    aberrations_mn = np.array(mn)
    aberrations_mn = aberrations_mn[np.argsort(aberrations_mn[:, 1]), :]

    sub = aberrations_mn[:, 1] > 0
    aberrations_mn[sub, :] = aberrations_mn[sub, :][
        np.argsort(aberrations_mn[sub, 0]), :
    ]
    aberrations_mn[~sub, :] = aberrations_mn[~sub, :][
        np.argsort(aberrations_mn[~sub, 0]), :
    ]
    aberrations_num = aberrations_mn.shape[0]

    sx, sy = intesity_size
    dx, dy = sampling_size
    wavelength = electron_wavelength_angstrom(energy)

    qx = xp.fft.fftfreq(sx, dx)
    qy = xp.fft.fftfreq(sy, dy)
    qr2 = qx[:, None] ** 2 + qy[None, :] ** 2
    alpha = xp.sqrt(qr2) * wavelength
    theta = xp.arctan2(qy[None, :], qx[:, None])

    # Aberration basis
    aberrations_basis = xp.ones((alpha.size, aberrations_num), dtype=xp.float32)

    # Skip constant to avoid dividing by zero in normalization
    for a0 in range(0, aberrations_num):
        m, n, a = aberrations_mn[a0]
        if n == 0:
            # Radially symmetric basis
            aberrations_basis[:, a0] = (alpha ** (m + 1) / (m + 1)).ravel()

        elif a == 0:
            # cos coef
            aberrations_basis[:, a0] = (
                alpha ** (m + 1) * xp.cos(n * theta) / (m + 1)
            ).ravel()
        else:
            # sin coef
            aberrations_basis[:, a0] = (
                alpha ** (m + 1) * xp.sin(n * theta) / (m + 1)
            ).ravel()

    # global scaling
    aberrations_basis *= 2 * np.pi / wavelength

    return aberrations_basis, aberrations_mn


# def image_shift_basis_function(
#     intesity_size,
#     sampling_size,
#     energy,
#     xp=np,
# ):

#     sx, sy = intesity_size
#     dx, dy = sampling_size
#     wavelength = electron_wavelength_angstrom(energy)

#     qx = xp.fft.fftfreq(sx, dx)
#     qy = xp.fft.fftfreq(sy, dy)
#     qr2 = qx[:, None] ** 2 + qy[None, :] ** 2
#     alpha = xp.sqrt(qr2) * wavelength
#     theta = xp.arctan2(qy[None, :], qx[:, None])

#     # Aberration basis
#     image_shift_basis = xp.ones((alpha.size, 2))

#     # cos coef
#     image_shift_basis[:, 0] = (
#         alpha ** (0 + 1) * xp.cos(1 * theta) / (0 + 1)
#     ).ravel()
    
#     # sin coef
#     image_shift_basis[:, 1] = (
#         alpha ** (0 + 1) * xp.sin(1 * theta) / (0 + 1)
#     ).ravel()

#     # global scaling
#     image_shift_basis *= 2 * np.pi / wavelength

#     return image_shift_basis

def fit_aberration_surface(
    chi_function,
    sampling,
    energy,
    max_angular_order,
    min_radial_order,
    max_radial_order,
    xp=np,
):
    """ """

    raveled_basis, _ = aberrations_basis_function(
        chi_function.shape,
        sampling,
        energy,
        max_angular_order,
        min_radial_order,
        max_radial_order,
        xp=xp,
    )

    Aw = raveled_basis 
    bw = chi_function.ravel() 
    coeff = xp.linalg.lstsq(Aw, bw, rcond=None)[0]

    fitted_chi_function = xp.tensordot(raveled_basis, coeff, axes=1).reshape(chi_function.shape)

    return fitted_chi_function, coeff


def gradient_strengh_correction(
    object_update,
    sampling_size,
    energy,
    xp=np,
):
    """ """
    sx, sy = object_update.shape
    dx, dy = sampling_size
    wavelength = electron_wavelength_angstrom(energy)

    qx = xp.fft.fftfreq(sx, dx)
    qy = xp.fft.fftfreq(sy, dy)
    qr2 = qx[:, None] ** 2 + qy[None, :] ** 2
    freq_correction = 1/2+xp.exp(-qr2)/2 #* 

    object_update = xp.real(xp.fft.ifft2(xp.fft.fft2(object_update)*freq_correction)).astype(xp.float32)

    return object_update
