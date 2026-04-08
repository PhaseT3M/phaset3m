# Juhyeok Lee, LBNL, 2023.
# This code is based on py4dstem (especially referring to overlap tomography part)
# 03.27.2025, major update (by Juhyeok Lee): reconstruction update

import warnings
from typing import Mapping, Sequence, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PhaseT3M.process.tqdmnd import tqdmnd

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
import os

from PhaseT3M.datastack.datastack import DataStack
from PhaseT3M.process.iterative_contraints import Contraints
from PhaseT3M.process.iterative_methods import Reconstruction_methods
from PhaseT3M.process.visualize_tools import Visualize_tools
from PhaseT3M.process.rotation import Image3DRotation

from PhaseT3M.process.utils import (electron_wavelength_angstrom,
                                    aberrations_basis_function,
                                    )



warnings.simplefilter(action="always", category=UserWarning)

class TomographicReconstruction(
    Contraints,
    Reconstruction_methods,
    Visualize_tools):

    def __init__(
        self,
        energy: float,
        pixel_size: float,
        num_slices: int,
        tilt_orientation_matrices: Sequence[np.ndarray],
        datastack: Sequence[DataStack] = None,
        object_type: str = "potential",
        initial_object_guess: np.ndarray = None,
        incident_wave_guess: np.ndarray = None,
        object_padding_px: Tuple[int, int] = [0, 0],
        verbose: bool = True,
        device: str = "cpu",
        name: str = "tomographic_reconstruction",
        **kwargs,
    ):

        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
            from scipy.ndimage import affine_transform, gaussian_filter, rotate, zoom, grey_dilation

            self._gaussian_filter = gaussian_filter
            self._zoom = zoom
            self._rotate = rotate
            self._grey_dilation = grey_dilation
            self._affine_transform = affine_transform
            from scipy.special import erf

            self._erf = erf
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
            from cupyx.scipy.ndimage import affine_transform, gaussian_filter, rotate, zoom, grey_dilation

            self._gaussian_filter = gaussian_filter
            self._zoom = zoom
            self._rotate = rotate
            self._grey_dilation = grey_dilation
            self._affine_transform = affine_transform
            from cupyx.scipy.special import erf

            self._erf = erf
        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")    
        

        self._datastack = datastack
        self._object = initial_object_guess
        self._incident_wave = incident_wave_guess

        # Common Metadata
        self._energy = energy
        self._pixel_sizes = np.array([pixel_size, pixel_size])
        self._object_type = object_type
        self._object_padding_px = object_padding_px
        self._verbose = verbose
        self._device = device
        self._preprocessed = False
        self._wavelength = electron_wavelength_angstrom(energy)

        # Class-specific Metadata
        self._num_slices = num_slices
        self._tilt_orientation_matrices = np.array(tilt_orientation_matrices)
        self._num_tilts = len(tilt_orientation_matrices)


    def preprocess(
        self,
        reconstruction_rotation_method: Literal["interpolation", "Fourier_sheer"] = "interpolation",
        object_optimizer: Literal["GD", "rmsprop"] = "GD",
        aberration_optimizer: Literal["GD", "adam"] = "adam",
        amplitude_normalization: bool = True,
        aberrations_max_angular_order: int = 1,
        aberrations_max_radial_order: int = 2,
        defocus_spread: float = 0, # Cc/(delta E/ E)
        progress_bar: bool = True,
        **kwargs,
    ):
        
        xp = self._xp
        asnumpy = self._asnumpy

        # set additional metadata
        self._reconstruction_rotation_method = reconstruction_rotation_method
        self._object_optimizer = object_optimizer
        self._aberration_optimizer = aberration_optimizer

        if self._datastack is None:
            raise ValueError(
                (
                    "The preprocess() method requires a DataStack. "
                )
            )
        
        if self._datastack[0].data.ndim != 3:
            raise ValueError(
                (
                    "The DataStack format should be (Number of focal series, Image size Y, Image size X)."
                )
            )

        # Object Initialization
        if self._object is None:
            size_x = self._datastack[0].data.shape[2]
            size_y = self._datastack[0].data.shape[1]
            
            if self._object_type == 'complex':
                self._object = xp.zeros((size_x, size_y, size_x), dtype=xp.complex64)
            else:
                self._object = xp.zeros((size_x, size_y, size_x), dtype=xp.float32)
        else:
            if self._object_type == 'complex':
                self._object = xp.asarray(self._object, dtype=xp.complex64)
            else:
                self._object = xp.asarray(self._object, dtype=xp.float32)

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[0]       
        self._num_defocus = np.max([len(self._datastack[indx].defocus) for indx in range(len(self._datastack))])
        self._padded_px = (self._object_shape[0]+2*self._object_padding_px[0],self._object_shape[1]+2*self._object_padding_px[1])

        # Tilt orientation matrices initialization
        self._tilt_orientation_matrices_initial = self._tilt_orientation_matrices.copy()

        # Precomputed propagator arrays
        self.sampling = self._pixel_sizes
        self._slice_thicknesses = np.tile(
            self._object_shape[1] * self.sampling[1] / self._num_slices,
            self._num_slices - 1,
        )
        self._propagator_arrays = self._precompute_propagator_arrays(
            self._padded_px,
            self.sampling,
            self._energy,
            self._slice_thicknesses,
        )
        
        # Incident wave Initialization
        if self._incident_wave is None:
            self._incident_wave = xp.ones([self._num_tilts,self._object_shape[0],self._object_shape[1]], dtype=xp.complex64)
        
        # Normalize initial incident wave
        self._incident_wave = (self._incident_wave \
                            / xp.sqrt(xp.sum(xp.abs(self._incident_wave)**2, axis = (1,2))).reshape(-1,1,1) \
                            * xp.sqrt(self._incident_wave.shape[1]*self._incident_wave.shape[2])).astype(xp.complex64)

        # predicted wave
        self._predicted_exit_waves = xp.ones([self._num_tilts,self._object_shape[0],self._object_shape[1]], dtype=xp.complex64)

        if self._object_padding_px != [0, 0]:
            # padded initial incident wave
            self._incident_wave = xp.pad(self._incident_wave,
                                    ((0, 0), (self._object_padding_px[0], self._object_padding_px[0]),(self._object_padding_px[1], self._object_padding_px[1]),),
                                    mode="constant", constant_values=1).astype(xp.complex64)

            # padded predicted wave
            self._predicted_exit_waves = xp.ones([self._num_tilts,self._padded_px[0],self._padded_px[1]], dtype=xp.complex64)

        self._incident_wave_initial = self._incident_wave.copy()
        self._cropping_px = (self._object_padding_px[0],self._object_padding_px[0]+self._object_shape[0],self._object_padding_px[1],self._object_padding_px[1]+self._object_shape[1])
        


        # extract intensities and normalize
        self._amplitudes = xp.empty([self._num_tilts,self._num_defocus,self._object_shape[0],self._object_shape[1]]
                                    , dtype=xp.float32)
        
        self._aberrations_basis, self._aberrations_mn = aberrations_basis_function(
                                                self._padded_px,
                                                self.sampling,
                                                self._energy,
                                                max_angular_order=aberrations_max_angular_order,
                                                min_radial_order=1,
                                                max_radial_order=aberrations_max_radial_order,
                                                xp = xp,
                                                )
        self._image_shift_basis, _ = aberrations_basis_function(
                                                self._padded_px,
                                                self.sampling,
                                                self._energy,
                                                max_angular_order=1,
                                                min_radial_order=0,
                                                max_radial_order=1,
                                                xp = xp,
                                                )

        self._cropped_image_shift_basis, _ = aberrations_basis_function(
                                                self._object_shape,
                                                self.sampling,
                                                self._energy,
                                                max_angular_order=1,
                                                min_radial_order=0,
                                                max_radial_order=1,
                                                xp = xp,
                                                )

        self._aberrations_coefs = xp.zeros([self._num_tilts, self._num_defocus, self._aberrations_basis.shape[1]])
        self._image_shift_coefs = xp.zeros([self._num_tilts, self._num_defocus, self._image_shift_basis.shape[1]])
        self._C1_ind = np.argmin(np.abs(self._aberrations_mn[:, 0] - 1.0) + self._aberrations_mn[:, 1])

        # create chi function
        self._chi_function = xp.zeros([self._num_tilts, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
        
        # create temperal coherence envelop function (treatment of temperal coherence)
        self._temperal_coherence_envelop_function = xp.exp(-1/4*(defocus_spread)**2 * (self._aberrations_basis.T[self._C1_ind]/2)**2).reshape([self._padded_px[0], self._padded_px[1]])
        
        # optimizer data
        if self._object_optimizer == 'rmsprop':
            self._object_grad_s = xp.zeros([self._num_tilts, self._num_slices], dtype=xp.float32)
            self._object_grad_s_initial = self._object_grad_s.copy()

        if self._aberration_optimizer == 'adam':
            self._aberrations_coefs_m = xp.zeros([self._num_tilts, self._num_defocus, self._aberrations_basis.shape[1]], dtype=xp.float32)
            self._aberrations_coefs_v = xp.zeros([self._num_tilts, self._num_defocus, self._aberrations_basis.shape[1]], dtype=xp.float32)
            self._aberrations_coefs_m_initial = self._aberrations_coefs_m.copy()
            self._aberrations_coefs_v_initial = self._aberrations_coefs_v.copy()

            self._image_shift_coefs_m = xp.zeros([self._num_tilts, self._num_defocus, self._image_shift_basis.shape[1]], dtype=xp.float32)
            self._image_shift_coefs_v = xp.zeros([self._num_tilts, self._num_defocus, self._image_shift_basis.shape[1]], dtype=xp.float32)
            self._image_shift_coefs_m_initial = self._image_shift_coefs_m.copy()
            self._image_shift_coefs_v_initial = self._image_shift_coefs_v.copy()

            self._chi_function_m = xp.zeros([self._num_tilts, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._chi_function_v = xp.zeros([self._num_tilts, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._chi_function_m_initial = self._chi_function_m.copy()
            self._chi_function_v_initial = self._chi_function_v.copy()

        for tilt_index in tqdmnd(
            self._num_tilts,
            desc="Preprocessing data",
            unit="tilt",
            disable=not progress_bar,
        ):
            self._amplitudes[tilt_index] = self._normalize_intensities(self._datastack[tilt_index].data, amplitude_normalization = amplitude_normalization)
            self._aberrations_coefs[tilt_index, :, self._C1_ind] = xp.array(-1*self._datastack[tilt_index].defocus)
            self._chi_function[tilt_index] = (xp.matmul(self._aberrations_coefs[tilt_index], self._aberrations_basis.T)
                                                +xp.matmul(self._image_shift_coefs[tilt_index], self._image_shift_basis.T)
                                                ).reshape(-1, self._padded_px[0], self._padded_px[1]).astype(xp.float32)

        self._aberrations_coefs_initial = self._aberrations_coefs.copy()
        self._image_shift_coefs_initial = self._image_shift_coefs.copy()
        self._chi_function_initial = self._chi_function.copy()

        # 3D rotation class
        self._rotate3d = Image3DRotation(shape=self._object.shape, rot_method = self._reconstruction_rotation_method, object_type = self._object_type, xp=self._xp, MEMORY_MAX_DIM= 600*600*600)

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self   
            

    def reconstruct(
        self,
        num_iter: int = 20,
        #reconstruction_method: str = "gradient-descent",
        seed_random: int = None,
        step_size: float = 0.01,
        normalization_min: float = 1,
        fix_chi_func_iter: int = np.inf,
        chi_func_step_size: float = 0.0000001,
        fix_image_shift_iter: int = np.inf,
        image_shift_step_size: float = 0.0000001,
        fix_aberrations_coefs_iter: int = np.inf,
        aberrations_step_size: float = 0.00001,
        fix_incident_wave_iter: int = np.inf,
        butterworth_filter_iter: int = np.inf,
        q_lowpass: float = None,
        q_highpass: float = None,
        butterworth_order: float = 2,
        butterworth_3dfilter_iter: int = np.inf,
        q_lowpass_3d: float = None,
        q_highpass_3d: float = None,
        butterworth_order_3d: float = 2,
        object_positivity: bool = True,
        object_imag_positivity: bool = False,
        object_3d_positivity: bool = True,
        object_imag_3d_positivity: bool = False,
        object_threshold_iter: int = np.inf,
        object_threshold_val: float = 0.0,
        object_3dmask: np.ndarray = None,
        denoise_tv_chambolle_iter: int = 0,
        denoise_tv_weight: float = 0.1,
        denoise_tv_axis: int = None,
        collective_tilt_updates: bool = False,
        store_iterations: bool = False,
        progress_bar: bool = True,
        reset: bool = None,
        **kwargs,
    ):
        """
        Tomographic reconstruction main method.

        Parameters
        --------
        num_iter: int, optional
            Maximum number of iterations to run
        # reconstruction_method: str, optional
        #     Specifies which reconstruction algorithm to use, one of:
        #     "GD" (or "gradient_descent")
        #     ...further development later.. 
        seed_random : int, optional
            Seed for the random number generator.
        step_size : float, optional
            Step size for updating the 3D object.
        normalization_min : float, optional
            Minimum probe normalization as a fraction of maximum overlap intensity.
        fix_chi_func_iter : int, optional
            Number of iterations with fixed positions before updating the chi function.
        chi_func_step_size : float, optional
            Step size for updating the chi function.
        fix_image_shift_iter : int, optional
            Number of iterations with fixed positions before updating image shift.
        image_shift_step_size : float, optional
            Step size for updating the image shift.
        fix_aberrations_coefs_iter : int, optional
            Number of iterations with fixed positions before updating aberration coefficients.
        aberrations_step_size : float, optional
            Step size for updating aberration coefficients.
        fix_incident_wave_iter : int, optional
            Number of iterations with fixed positions before updating the incident wave function.
        butterworth_filter_iter : int, optional
            Number of iterations using a 2D Butterworth filter.
        q_lowpass : float
            Low-pass cutoff frequency (in Å⁻¹) for the 2D Butterworth filter.
        q_highpass : float
            High-pass cutoff frequency (in Å⁻¹) for the 2D Butterworth filter.
        butterworth_order : float
            Order of the 2D Butterworth filter; lower values result in smoother filters.
        butterworth_3dfilter_iter : int, optional
            Number of iterations using a 3D Butterworth filter.
        q_lowpass_3d : float
            Low-pass cutoff frequency (in Å⁻¹) for the 3D Butterworth filter.
        q_highpass_3d : float
            High-pass cutoff frequency (in Å⁻¹) for the 3D Butterworth filter.
        butterworth_order_3d : float
            Order of the 3D Butterworth filter; lower values result in smoother filters.
        object_positivity : bool, optional
            If True, enforces positivity on the object.
        object_imag_positivity : bool, optional
            If True, enforces positivity on the imaginary part of the object.
        object_threshold_iter : int, optional
            Number of iterations to apply phase thresholding.
        object_threshold_val : float
            Phase shift (in radians) subtracted from the potential at each iteration.
        object_3dmask : np.array
            3D mask to apply to the object, if specified.
        denoise_tv_chambolle_iter : int, optional
            Number of iterations for TV denoising.
        denoise_tv_weight : float
            Weight parameter for TV denoising.
        denoise_tv_axis : int, optional
            Axis for TV denoising.
        collective_tilt_updates : bool, optional
            If True, updates the object collectively with tilt corrections.
        store_iterations : bool, optional
            If True, stores reconstructed objects and probes at each iteration.
        progress_bar : bool, optional
            If True, displays reconstruction progress.
        reset : bool, optional
            If True, ignores previous reconstructions and resets.

        Returns
        -------
        self : TomographicReconstruction
            The instance of TomographicReconstruction for method chaining.
        """

        asnumpy = self._asnumpy
        xp = self._xp

        # initialization
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.incident_wave_iterations = []
            self.chi_function_iterations = []
            self.predicted_exit_wave_iterations = []

        if reset:
            self._object = self._object_initial.copy()
            self._rot_object = self._object_initial.copy()
            self._incident_wave = self._incident_wave_initial.copy()
            self._aberrations_coefs = self._aberrations_coefs_initial.copy()
            self._image_shift_coefs = self._image_shift_coefs_initial.copy()
            self._chi_function = self._chi_function_initial.copy()
            self.error_iterations = []

            if self._object_optimizer == 'rmsprop':
                self._object_grad_s = self._object_grad_s_initial.copy()

            if self._aberration_optimizer == 'adam':
                self._aberrations_coefs_m = self._aberrations_coefs_m_initial.copy()
                self._aberrations_coefs_v = self._aberrations_coefs_v_initial.copy()
                self._image_shift_coefs_m = self._image_shift_coefs_m_initial.copy()
                self._image_shift_coefs_v = self._image_shift_coefs_v_initial.copy()
                self._chi_function_m = self._chi_function_m_initial.copy()
                self._chi_function_v = self._chi_function_v_initial.copy()

            self._residual_waves = None

        elif reset is None:
            if hasattr(self, "error"):
                warnings.warn(
                    (
                        "Continuing reconstruction from previous result. "
                        "Use reset=True for a fresh start."
                    ),
                    UserWarning,
                )
            else:
                self.error_iterations = []
                self._residual_waves = None

        np.random.seed(seed_random)
        
        # main loop
        for a0 in tqdmnd(
            num_iter,
            desc="Reconstructing object",
            unit=" iter",
            disable=not progress_bar,
        ):
            self._active_iter = a0

            error = 0.0

            if collective_tilt_updates:
                collective_object = xp.zeros_like(self._object)
                
            tilt_indices = np.arange(self._num_tilts)
            np.random.shuffle(tilt_indices)


            for tilt_index in tilt_indices:
                
                self._active_tilt_index = tilt_index

                tilt_error = 0.0

                # 3D rotation
                self._rot_object = self._object.copy() # rotated object
                rot_matrix = self._tilt_orientation_matrices[self._active_tilt_index]
                if np.sum(np.abs(rot_matrix-np.eye(3))) > 0.001: # Rotations smaller than ~0.05 deg. are ignored.
                    self._rot_object = self._rotate3d.rotate_3d(self._rot_object, rot_matrix)

                object_sliced = self._project_sliced_object(
                    self._rot_object, self._num_slices
                )
                object_sliced_old = object_sliced.copy()

                # pad
                object_sliced = xp.pad(object_sliced, ((0,0) \
                                                        ,(self._object_padding_px[0],self._object_padding_px[0]) \
                                                        ,(self._object_padding_px[1],self._object_padding_px[1])))
                    
                if a0 < fix_chi_func_iter:
                    self._chi_function[self._active_tilt_index] = (xp.matmul(self._aberrations_coefs[self._active_tilt_index], self._aberrations_basis.T) 
                                                                +xp.matmul(self._image_shift_coefs[self._active_tilt_index], self._image_shift_basis.T)
                                                                ).reshape(-1, self._padded_px[0], self._padded_px[1])
                    
                # forward
                (
                    propagated_waves,
                    complex_object,
                    predicted_detector_waves,
                    self._residual_waves,
                    tilt_error,
                ) = self._forward(
                    object_sliced,
                    self._incident_wave[tilt_index],
                    self._amplitudes[tilt_index],
                )

                # self.predicted_detector_waves = predicted_detector_waves
                
                # adjoint operator
                object_sliced, self._incident_wave[tilt_index] = self._adjoint(
                    current_object=object_sliced,
                    current_incident_wave=self._incident_wave[tilt_index],
                    complex_object=complex_object,
                    propagated_waves=propagated_waves,
                    residual_waves=self._residual_waves,
                    predicted_detector_waves=predicted_detector_waves,
                    fix_chi_func= a0 < fix_chi_func_iter,
                    fix_aberrations_coefs= a0 < fix_aberrations_coefs_iter,
                    aberrations_coefs=self._aberrations_coefs[tilt_index],
                    fix_image_shift_coefs= a0 < fix_image_shift_iter,
                    image_shift_coefs=self._image_shift_coefs[tilt_index],
                    step_size=step_size,
                    chi_func_step_size=chi_func_step_size,
                    aberrations_step_size=aberrations_step_size,
                    image_shift_step_size= image_shift_step_size,
                    normalization_min=normalization_min,
                    fix_incident_wave= a0 < fix_incident_wave_iter,
                )
                
                # crop
                object_sliced = object_sliced[:,self._cropping_px[0]:self._cropping_px[1],self._cropping_px[2]:self._cropping_px[3]]
                object_sliced -= object_sliced_old

                object_update = self._expand_sliced_object(
                    object_sliced, self._num_voxels
                )
                
                # 3D rotation back
                if np.sum(np.abs(rot_matrix-np.eye(3))) > 0.001: # Rotations smaller than ~0.05 deg. are ignored.
                    object_update = self._rotate3d.rotate_3d(object_update, rot_matrix.T)

                # object_update
                if collective_tilt_updates:
                    collective_object += object_update
                else:
                    self._object += object_update

                    # Contraints
                    (
                        self._object,
                        self._chi_function[self._active_tilt_index]
                    )= self._constraints(
                        current_object=self._object,
                        current_chi_function=self._chi_function[self._active_tilt_index],
                        butterworth_filter=a0 < butterworth_filter_iter
                        and (q_lowpass is not None or q_highpass is not None),
                        q_lowpass=q_lowpass,
                        q_highpass=q_highpass,
                        butterworth_order=butterworth_order,
                        object_positivity=object_positivity,
                        object_threshold= a0 >= object_threshold_iter,
                        object_threshold_val = object_threshold_val,
                        object_imag_positivity=object_imag_positivity,
                    )
                    

                # Normalize Error
                tilt_error /= (
                    self._object_shape[0]*self._object_shape[1]
                    * self._num_defocus
                )
                error += tilt_error
            

            # 3D Contraints
            self._object = self._3d_constraints(
                                    self._object,
                                    object_3dmask = object_3dmask,
                                    butterworth_3dfilter = a0 < butterworth_3dfilter_iter and (q_lowpass_3d is not None or q_highpass_3d is not None),
                                    q_lowpass_3d = q_lowpass_3d,
                                    q_highpass_3d = q_highpass_3d,
                                    butterworth_order_3d = butterworth_order_3d,
                                    object_positivity = object_3d_positivity,
                                    object_imag_positivity = object_imag_3d_positivity,
                                    denoise_tv_chambolle = a0 < denoise_tv_chambolle_iter,
                                    denoise_tv_weight = denoise_tv_weight,
                                    denoise_tv_axis = denoise_tv_axis,
                                )


            # Normalize Error Over Tilts
            error /= self._num_tilts

            if collective_tilt_updates:
                self._object += collective_object / self._num_tilts
                
                # Contraints
                (
                    self._object,
                    self._chi_function[self._active_tilt_index]
                )= self._constraints(
                    current_object=self._object,
                    current_chi_function=self._chi_function[self._active_tilt_index],
                    butterworth_filter=a0 < butterworth_filter_iter
                    and (q_lowpass is not None or q_highpass is not None),
                    q_lowpass=q_lowpass,
                    q_highpass=q_highpass,
                    butterworth_order=butterworth_order,
                    object_positivity=object_positivity,
                    object_threshold= a0 >= object_threshold_iter,
                    object_threshold_val = object_threshold_val,
                    object_imag_positivity=object_imag_positivity,
                )
                # 3D Contraints
                self._object = self._3d_constraints(
                        self._object,
                        object_3dmask = object_3dmask,
                        butterworth_3dfilter = a0 < butterworth_3dfilter_iter and (q_lowpass_3d is not None or q_highpass_3d is not None),
                        q_lowpass_3d = q_lowpass_3d,
                        q_highpass_3d = q_highpass_3d,
                        butterworth_order_3d = butterworth_order_3d,
                        object_positivity = object_3d_positivity,
                        object_imag_positivity = object_imag_3d_positivity,
                        denoise_tv_chambolle = a0 < denoise_tv_chambolle_iter,
                        denoise_tv_weight = denoise_tv_weight,
                        denoise_tv_axis = denoise_tv_axis,
                    )

            self.error_iterations.append(error.item())

            if store_iterations:
                self.object_iterations.append(asnumpy(self._object.copy()))
                self.incident_wave_iterations.append(asnumpy(self._incident_wave.copy()))
                self.chi_function_iterations.append(asnumpy(self._chi_function.copy()))
                self.predicted_exit_wave_iterations.append(asnumpy(self._predicted_exit_waves.copy()))

        # store result
        self.object = asnumpy(self._object)
        self.incident_wave = asnumpy(self._incident_wave)
        self.chi_function = asnumpy(self._chi_function)
        self.predicted_exit_waves = asnumpy(self._predicted_exit_waves)
        self.error = error.item()

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self


    def visualize(
        self,
        fig=None,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        plot_phase_aberration_function: bool = True,
        plot_abs_aberration_function: bool = False,
        plot_nm_aberration_function: Tuple[int, int] = (0, 0),
        plot_imaginary_object: bool = False,
        cbar: bool = True,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims=(None, None),
        y_lims=(None, None),
        **kwargs,
    ):
        """
        Displays reconstructed object and probe.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_phase_aberration_function: bool
            If true, the phase aberration_function is also displayed
        plot_abs_aberration_function: bool, optional
            If true, the abs aberration_function is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices

        Returns
        --------
        self: TomographicReconstruction
            Self to accommodate chaining
        """

        if iterations_grid is None:
            self._visualize_last_iteration(
                fig=fig,
                plot_convergence=plot_convergence,
                plot_phase_aberration_function=plot_phase_aberration_function,
                plot_abs_aberration_function=plot_abs_aberration_function,
                plot_nm_aberration_function=plot_nm_aberration_function,
                plot_imaginary_object=plot_imaginary_object,
                cbar=cbar,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
                **kwargs,
            )
        else:
            self._visualize_all_iterations(
                fig=fig,
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                plot_phase_aberration_function=plot_phase_aberration_function,
                plot_abs_aberration_function=plot_abs_aberration_function,
                plot_nm_aberration_function=plot_nm_aberration_function,
                plot_imaginary_object=plot_imaginary_object,
                cbar=cbar,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
                **kwargs,
            )

        return self

    def _return_object_fft(
        self,
        obj=None,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims: Tuple[int, int] = (None, None),
        y_lims: Tuple[int, int] = (None, None),
    ):
        """
        Returns obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if obj is None:
            obj = self._object.real
        else:
            obj = xp.asarray(obj.real, dtype=xp.float32)

        if projection_angle_deg is not None:
            rotated_3d_obj = self._rotate(
                obj,
                projection_angle_deg,
                axes=projection_axes,
                reshape=False,
                order=2,
            )
            rotated_3d_obj = asnumpy(rotated_3d_obj)
        else:
            rotated_3d_obj = asnumpy(obj)

        rotated_object = self._crop_rotate_object_manually(
            rotated_3d_obj.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
        )

        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rotated_object))))

    def show_object_fft(
        self,
        obj=None,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims: Tuple[int, int] = (None, None),
        y_lims: Tuple[int, int] = (None, None),
        **kwargs,
    ):
        """
        Plot FFT of reconstructed object

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        if obj is None:
            object_fft = self._return_object_fft(
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
            )
        else:
            object_fft = self._return_object_fft(
                obj,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
            )

        figsize = kwargs.pop("figsize", (6, 6))
        cmap = kwargs.pop("cmap", "magma")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 1)
        power = kwargs.pop("power", 0.2)

        pixelsize = 1 / (object_fft.shape[0] * self.sampling[0])
        show(
            object_fft,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            scalebar=True,
            pixelsize=pixelsize,
            ticks=False,
            pixelunits=r"$\AA^{-1}$",
            power=power,
            **kwargs,
        )





