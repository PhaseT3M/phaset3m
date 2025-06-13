# Juhyeok Lee, LBNL, 2023.
# This code is based on py4dstem

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from scipy.ndimage import rotate as rotate_np
from PhaseT3M.process.tqdmnd import tqdmnd

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
    import os

from PhaseT3M.datastack.datastack import DataStack
from PhaseT3M.process.utils import (electron_wavelength_angstrom,
                              spatial_frequencies)

warnings.simplefilter(action="always", category=UserWarning)

class Contraints():
    def _constraints(
        self,
        current_object,
        current_chi_function,
        butterworth_filter: bool,
        q_lowpass,
        q_highpass,
        butterworth_order,
        object_positivity: bool,
        object_imag_positivity: bool,
        object_threshold: bool,
        object_threshold_val: float,
    ):
        """
        Reconstruction constraints operator.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_chi_function: np.ndarray
            Current chi function
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        object_positivity: bool
            If True, forces object to be positive

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        constrained_chi_function: np.ndarray
            Constrained chi function estimate
        """

        if object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        if object_imag_positivity:
            current_object = self._object_imaginary_positivity_constraint(current_object)
        
        if object_threshold:
            current_object = self._object_threshold_constraint(current_object, object_threshold_val)

        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
            current_object,
            q_lowpass,
            q_highpass,
            butterworth_order,
        )
            
        return current_object, current_chi_function
    


    def _3d_constraints(
        self,
        current_object,
        object_3dmask,
        butterworth_3dfilter: bool,
        q_lowpass_3d,
        q_highpass_3d,
        butterworth_order_3d,
        object_positivity: bool,
        object_imag_positivity: bool,
        denoise_tv_chambolle: bool,
        denoise_tv_weight: float,
        denoise_tv_axis: int,
    ):
        """
        Reconstruction 3d constraints operator.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        object_3dmask: np.ndarray
            3D mask array
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        object_positivity: bool
            If True, forces object to be positive

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        constrained_chi_function: np.ndarray
            Constrained chi function estimate
        """

        if object_3dmask is not None:
            object_3dmask = self._xp.asarray(object_3dmask, dtype=self._xp.float32)
            current_object = current_object*object_3dmask
        
        if butterworth_3dfilter:
            current_object = self._object3d_butterworth_constraint(
                                        current_object,
                                        q_lowpass_3d,
                                        q_highpass_3d,
                                        butterworth_order_3d,
                                    )

        if denoise_tv_chambolle:
            current_object = self._object_denoise_tv_chambolle(
                                        current_object,
                                        weight = denoise_tv_weight,
                                        axis = denoise_tv_axis,
                                        padding = None,
                                        eps=2.0e-4,
                                        max_num_iter=200,
                                        scaling=None,
                                    )
            
        if object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        if object_imag_positivity:
            current_object = self._object_imaginary_positivity_constraint(current_object)
            
        return current_object


    def _object_positivity_constraint(self, current_object):
        """
        Positivity constraint.
        Used to ensure potential is positive.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        if xp.iscomplexobj(current_object):
            return xp.maximum(current_object.real, 0.0) + 1j*current_object.imag
        else:
            return xp.maximum(current_object, 0.0)
        
        
    def _object_imaginary_positivity_constraint(self, current_object):
        """
        Positivity constraint.
        Used to ensure imaginary potential is positive.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        if xp.iscomplexobj(current_object):
            return current_object.real + 1j*xp.maximum(current_object.imag, 0.0)
        else:
            return current_object

    def _object_threshold_constraint(self, current_object, threshold_value):
        """
        Positivity constraint.
        Used to ensure potential is positive.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        if xp.iscomplexobj(current_object):
            tmp_object = current_object.real
            tmp_object[tmp_object<xp.max(current_object.real)*threshold_value] = 0
            return tmp_object + 1j*current_object.imag
        else:
            tmp_object = current_object
            tmp_object[tmp_object<xp.max(current_object)*threshold_value] = 0
            return tmp_object


    def _object_butterworth_constraint(
        self,
        current_object,
        q_lowpass,
        q_highpass,
        butterworth_order,
    ):
        """
        Butterworth filter.
        Used for low/high-pass filtering object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        qx = xp.fft.fftfreq(current_object.shape[2], self.sampling[1]).reshape(-1, 1)
        qy = xp.fft.fftfreq(current_object.shape[1], self.sampling[0]).reshape(1, -1)

        qra = xp.sqrt(qx**2 + qy**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object)
        current_object -= current_object_mean
        
        #current_object = xp.fft.ifft2(xp.fft.fft2(current_object) * env)
        boost_speed = 20
        for i in range(int(np.ceil(current_object.shape[0]/boost_speed))):
            if (i+1)*boost_speed < current_object.shape[0]:
                current_object[i*boost_speed:(i+1)*boost_speed] = xp.real(xp.fft.ifft2(xp.fft.fft2(current_object[i*boost_speed:(i+1)*boost_speed]) * env))
            else:
                current_object[i*boost_speed:current_object.shape[0]] = xp.real(xp.fft.ifft2(xp.fft.fft2(current_object[i*boost_speed:current_object.shape[0]]) * env))

        current_object += current_object_mean

        if self._object_type == "potential":
            current_object = xp.real(current_object)

        return current_object



    def _object3d_butterworth_constraint(
            self,
            current_object,
            q_lowpass,
            q_highpass,
            butterworth_order,
        ):
        xp = self._xp

        output_object = xp.asarray(current_object)

        #output_object = xp.pad(output_object, ((10, 10),(10, 10)))
        
        qx = xp.fft.fftfreq(output_object.shape[0], self.sampling[1]).reshape(-1, 1, 1)
        qy = xp.fft.fftfreq(output_object.shape[1], self.sampling[0]).reshape(1, -1, 1)
        qz = xp.fft.fftfreq(output_object.shape[2], self.sampling[1]).reshape(1, 1, -1)

        qra = xp.sqrt(qx**2 + qy**2 + qz**2)

        env = xp.ones_like(qra, dtype=xp.float32)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        output_object_mean = xp.mean(output_object)
        output_object -= output_object_mean
        
        #output_object = np.fft.fft(np.fft.fft(np.fft.fft(output_object, axis=0), axis=1), axis=2)
        output_object = xp.fft.ifftn(xp.fft.fftn(output_object) * env)
        
        output_object += output_object_mean
        
        if self._object_type == "potential":
            output_object = xp.real(output_object)

        #output_object = output_object#[10:output_object.shape[0]-10, 10:output_object.shape[1]-10]

        return output_object


    def _object_denoise_tv_chambolle(
        self,
        current_object0,
        weight,
        axis,
        padding,
        eps=2.0e-4,
        max_num_iter=200,
        scaling=None,
    ):
        """
        Perform total-variation denoising on n-dimensional images.

        Parameters
        ----------
        current_object: np.ndarray
            Current object estimate
        weight : float, optional
            Denoising weight. The greater `weight`, the more denoising (at
            the expense of fidelity to `input`).
        axis: int or tuple
            Axis for denoising, if None uses all axes
        pad_object: bool
            if True, pads object with zeros along axes of blurring
        eps : float, optional
            Relative difference of the value of the cost function that determines
            the stop criterion. The algorithm stops when:

                (E_(n-1) - E_n) < eps * E_0

        max_num_iter : int, optional
            Maximal number of iterations used for the optimization.
        scaling : tuple, optional
            Scale weight of tv denoise on different axes

        Returns
        -------
        constrained_object: np.ndarray
            Constrained object estimate

        Notes
        -----
        Rudin, Osher and Fatemi algorithm.
        Adapted skimage.restoration.denoise_tv_chambolle.
        """
        xp = self._xp

        if self._xp == cp:
            current_object = xp.asnumpy(current_object0).copy()
            xp = np

        if self._object_type == "complex":
            updated_object = current_object
            warnings.warn(
                (
                    "TV denoising is currently only supported for object_type=='potential'."
                ),
                UserWarning,
            )

        else:
            current_object_sum = xp.sum(current_object)

            if axis is None:
                ndim = xp.arange(current_object.ndim).tolist()
            elif isinstance(axis, tuple):
                ndim = list(axis)
            else:
                ndim = [axis]

            if padding is not None:
                pad_width = ((0, 0),) * current_object.ndim
                pad_width = list(pad_width)

                for ax in range(len(ndim)):
                    pad_width[ndim[ax]] = (padding, padding)

                current_object = xp.pad(
                    current_object, pad_width=pad_width, mode="constant"
                )

            p = xp.zeros(
                (current_object.ndim,) + current_object.shape,
                dtype=current_object.dtype,
            )
            g = xp.zeros_like(p)
            d = xp.zeros_like(current_object)

            i = 0
            while i < max_num_iter:
                if i > 0:
                    # d will be the (negative) divergence of p
                    d = -p.sum(0)
                    slices_d = [
                        slice(None),
                    ] * current_object.ndim
                    slices_p = [
                        slice(None),
                    ] * (current_object.ndim + 1)
                    for ax in range(len(ndim)):
                        slices_d[ndim[ax]] = slice(1, None)
                        slices_p[ndim[ax] + 1] = slice(0, -1)
                        slices_p[0] = ndim[ax]
                        d[tuple(slices_d)] += p[tuple(slices_p)]
                        slices_d[ndim[ax]] = slice(None)
                        slices_p[ndim[ax] + 1] = slice(None)
                    updated_object = current_object + d
                else:
                    updated_object = current_object
                E = (d**2).sum()

                # g stores the gradients of updated_object along each axis
                # e.g. g[0] is the first order finite difference along axis 0
                slices_g = [
                    slice(None),
                ] * (current_object.ndim + 1)
                for ax in range(len(ndim)):
                    slices_g[ndim[ax] + 1] = slice(0, -1)
                    slices_g[0] = ndim[ax]
                    g[tuple(slices_g)] = xp.diff(updated_object, axis=ndim[ax])
                    slices_g[ndim[ax] + 1] = slice(None)
                if scaling is not None:
                    scaling /= xp.max(scaling)
                    g *= xp.array(scaling)[:, xp.newaxis, xp.newaxis]
                norm = xp.sqrt((g**2).sum(axis=0))[xp.newaxis, ...]
                E += weight * norm.sum()
                tau = 1.0 / (2.0 * len(ndim))
                norm *= tau / weight
                norm += 1.0
                p -= tau * g
                p /= norm
                E /= float(current_object.size)
                if i == 0:
                    E_init = E
                    E_previous = E
                else:
                    if xp.abs(E_previous - E) < eps * E_init:
                        break
                    else:
                        E_previous = E
                i += 1

            if padding is not None:
                for ax in range(len(ndim)):
                    slices = array_slice(
                        ndim[ax], current_object.ndim, padding, -padding
                    )
                    updated_object = updated_object[slices]

            updated_object = (
                updated_object / xp.sum(updated_object) * current_object_sum
            )

        if self._xp == cp:
            updated_object = self._xp.array(updated_object)

        return updated_object