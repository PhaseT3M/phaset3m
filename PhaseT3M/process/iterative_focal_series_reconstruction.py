# Juhyeok Lee, LBNL, 2023.
# This code is based on py4dstem (especially referring to multislice ptychography part)
# 03.27.2025, major update (by Juhyeok Lee): reconstruction update

import warnings, time
from typing import Mapping, Sequence, Tuple, Union, Literal

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

class FocalSeriesReconstruction(
    Contraints,
    Reconstruction_methods,
    Visualize_tools):

    def __init__(
        self,
        energy: float,
        pixel_size: float,
        slice_thicknesses: Union[float, Sequence[float]] = 2.0,
        num_slices: int = 1,
        datastack: Sequence[DataStack] = None,
        object_type: str = "potential",
        initial_object_guess: np.ndarray = None,
        incident_wave_guess: np.ndarray = None,
        object_padding_px: Tuple[int, int] = [0, 0],
        verbose: bool = True,
        device: str = "cpu",
        name: str = "focal_series_reconstruction",
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

        slice_thicknesses = np.array(slice_thicknesses)
        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, num_slices - 1)
        elif slice_thicknesses.shape[0] != (num_slices - 1):
            raise ValueError(
                (
                    f"slice_thicknesses must have length {num_slices - 1}, "
                    f"not {slice_thicknesses.shape[0]}."
                )
            )
        self._slice_thicknesses = slice_thicknesses
        
        # Class-specific Metadata
        self._num_slices = num_slices



    def preprocess(
        self,
        object_optimizer: Literal["GD", "adam"] = "GD",
        aberration_optimizer: Literal["GD", "adam"] = "adam",
        amplitude_normalization: bool =  True,
        aberrations_max_angular_order: int = 1,
        aberrations_max_radial_order: int = 2,
        defocus_spread: float = 0.0, # Cc/(delta E/ E)
        progress_bar: bool = True,
        **kwargs,
    ):
        
        xp = self._xp
        asnumpy = self._asnumpy

        # set additional metadata
        self._object_optimizer = object_optimizer
        self._aberration_optimizer = aberration_optimizer

        if self._datastack is None:
            raise ValueError(
                (
                    "The preprocess() method requires a DataStack. "
                )
            )
        
        if self._datastack.data.ndim != 3:
            raise ValueError(
                (
                    "The DataStack format should be (Number of focal series, Image size Y, Image size X)."
                )
            )
        
        # Object Initialization
        if self._object is None:
            size_x = self._datastack.data.shape[2]
            size_y = self._datastack.data.shape[1]
            
            if self._object_type == 'complex':
                self._object = xp.zeros((self._num_slices, size_y, size_x), dtype=xp.complex64)
            else:
                self._object = xp.zeros((self._num_slices, size_y, size_x), dtype=xp.float32)
        else:
            if self._object_type == 'complex':
                self._object = xp.asarray(self._object, dtype=xp.complex64)
            else:
                self._object = xp.asarray(self._object, dtype=xp.float32)

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[0]       
        self._num_defocus = len(self._datastack.defocus)
        self._padded_px = (self._object_shape[0]+2*self._object_padding_px[0],self._object_shape[1]+2*self._object_padding_px[1])

        # Precomputed propagator arrays
        self.sampling = self._pixel_sizes
        # self._slice_thicknesses = np.tile(
        #     self._object_shape[1] * self.sampling[1] / self._num_slices,
        #     self._num_slices - 1,
        # )
        self._propagator_arrays = self._precompute_propagator_arrays(
            self._padded_px,
            self.sampling,
            self._energy,
            self._slice_thicknesses,
        )
        
        # Incident wave Initialization
        if self._incident_wave is None:
            self._incident_wave = xp.ones([1, self._object_shape[0],self._object_shape[1]], dtype=xp.complex64)
        
        # Normalize initial incident wave
        self._incident_wave = self._incident_wave \
                            / xp.sqrt(xp.sum(xp.abs(self._incident_wave)**2, axis = (1,2))).reshape(-1,1,1) \
                            * xp.sqrt(self._incident_wave.shape[1]*self._incident_wave.shape[2])

        # predicted wave
        self._predicted_exit_waves = xp.ones([1, self._object_shape[0],self._object_shape[1]], dtype=xp.complex64)

        if self._object_padding_px != [0, 0]:
            # padded initial incident wave
            self._incident_wave = xp.pad(self._incident_wave,
                                    ((0, 0), (self._object_padding_px[0], self._object_padding_px[0]),(self._object_padding_px[1], self._object_padding_px[1]),),
                                    mode="constant", constant_values=1).astype(xp.complex64)

            # padded predicted wave
            self._predicted_exit_waves = xp.ones([1,self._padded_px[0],self._padded_px[1]], dtype=xp.complex64)
        
        self._incident_wave_initial = self._incident_wave.copy()
        self._cropping_px = (self._object_padding_px[0],self._object_padding_px[0]+self._object_shape[0],self._object_padding_px[1],self._object_padding_px[1]+self._object_shape[1])
        


        # extract intensities and normalize
        self._amplitudes = xp.empty([1,self._num_defocus,self._object_shape[0],self._object_shape[1]]
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

        self._aberrations_coefs = xp.zeros([1, self._num_defocus, self._aberrations_basis.shape[1]], dtype=xp.float32)
        self._image_shift_coefs = xp.zeros([1, self._num_defocus, self._image_shift_basis.shape[1]], dtype=xp.float32)
        self._C1_ind = np.argmin(np.abs(self._aberrations_mn[:, 0] - 1.0) + self._aberrations_mn[:, 1])

        # create chi function
        self._chi_function = xp.zeros([1, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
        
        # create temperal coherence envelop function (treatment of temperal coherence)
        self._temperal_coherence_envelop_function = xp.exp(-1/4*(defocus_spread)**2 * (self._aberrations_basis.T[self._C1_ind]/2)**2).reshape([self._padded_px[0], self._padded_px[1]])
        
        # optimizer data
        if self._object_optimizer == 'adam':
            self._object_grad_m = xp.zeros([1, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._object_grad_v = xp.zeros([1, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._object_grad_m_initial = self._object_grad_m.copy()
            self._object_grad_v_initial = self._object_grad_v.copy()

        if self._aberration_optimizer == 'adam':
            self._aberrations_coefs_m = xp.zeros([1, self._num_defocus, self._aberrations_basis.shape[1]], dtype=xp.float32)
            self._aberrations_coefs_v = xp.zeros([1, self._num_defocus, self._aberrations_basis.shape[1]], dtype=xp.float32)
            self._aberrations_coefs_m_initial = self._aberrations_coefs_m.copy()
            self._aberrations_coefs_v_initial = self._aberrations_coefs_v.copy()

            self._image_shift_coefs_m = xp.zeros([1, self._num_defocus, self._image_shift_basis.shape[1]], dtype=xp.float32)
            self._image_shift_coefs_v = xp.zeros([1, self._num_defocus, self._image_shift_basis.shape[1]], dtype=xp.float32)
            self._image_shift_coefs_m_initial = self._image_shift_coefs_m.copy()
            self._image_shift_coefs_v_initial = self._image_shift_coefs_v.copy()

            self._chi_function_m = xp.zeros([1, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._chi_function_v = xp.zeros([1, self._num_defocus, self._padded_px[0], self._padded_px[1]], dtype=xp.float32)
            self._chi_function_m_initial = self._chi_function_m.copy()
            self._chi_function_v_initial = self._chi_function_v.copy()

        for tilt_index in tqdmnd(
            1,
            desc="Preprocessing data",
            unit="tilt",
            disable=not progress_bar,
        ):
            self._amplitudes[tilt_index] = self._normalize_intensities(self._datastack.data, amplitude_normalization = amplitude_normalization)
            self._aberrations_coefs[tilt_index, :, self._C1_ind] = xp.array(-1*self._datastack.defocus)
            self._chi_function[tilt_index] = (xp.matmul(self._aberrations_coefs[tilt_index], self._aberrations_basis.T)
                                                +xp.matmul(self._image_shift_coefs[tilt_index], self._image_shift_basis.T)
                                                ).reshape(-1, self._padded_px[0], self._padded_px[1]).astype(xp.float32)

        self._aberrations_coefs_initial = self._aberrations_coefs.copy()
        self._image_shift_coefs_initial = self._image_shift_coefs.copy()
        self._chi_function_initial = self._chi_function.copy()

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self   
            

    def reconstruct(
        self,
        num_iter: int = 20,
        # reconstruction_method: str = "gradient-descent",
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
        object_positivity: bool = True,
        object_imag_positivity: bool = False,
        object_threshold_iter: int = np.inf,
        object_threshold_val: float = 0.0,
        # denoise_tv_chambolle_iter: int = 0,
        # denoise_tv_weight: float = 0.1,
        # denoise_tv_axis: int = None,
        store_iterations: bool = False,
        progress_bar: bool = True,
        reset: bool = None,
        **kwargs,
    ):
        """
        Ptychographic reconstruction main method.

        Parameters
        --------
        num_iter: int, optional
            Maximum number of iterations to run
        # reconstruction_method: str, optional
        #     Specifies which reconstruction algorithm to use, one of:
        #     "GD" (or "gradient_descent")
        #     ...further development later.. 
        seed_random: int, optional
            Seeds the random number generator.
        step_size: float, optional
            Update step size of 3D object
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_chi_func_iter: int, optional
            Number of iterations to run with fixed positions before updating chi function in aberration
        chi_func_step_size: float, optional
            Update step size of chi function 
        fix_image_shift_iter: int, optional
            Number of iterations to run with fixed positions before updating image shift
        image_shift_step_size: float, optional
            Update step size of image shift
        fix_aberrations_coefs_iter: int, optional
            Number of iterations to run with fixed positions before updating aberration coefficients
        aberrations_step_size: float, optional
            Update step size of aberration coefficients
        fix_incident_wave_iter: int, optional
            Number of iterations to run with fixed positions before updating incident wave function
        butterworth_filter_iter: int, optional
            Number of iterations to run using 2D butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for 2D low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for 2D high-pass butterworth filter
        butterworth_order: float
            2D Butterworth filter order. Smaller gives a smoother filter
        object_positivity: bool, optional
            If True, forces object to be positive
        object_imag_positivity: bool, optional
            If True, forces imaginary part of object to be positive
        object_threshold_iter: int, optional
            If True, Phase shift in radians to be subtracted from the potential at each iteration
        object_threshold_val: float 
            Phase shift in radians to be subtracted from the potential at each iteration
        denoise_tv_chambolle_iter: int, optional
            Number of iterations to apply TV denoising
        denoise_tv_weight: float 
            Weight parameter of TV denoising
        denoise_tv_axis: int, optional
            Axis of TV denoising
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        reset: bool, optional
            If True, previous reconstructions are ignored

        Returns
        --------
        self: FocalSeriesReconstruction
            Self to accommodate chaining
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
            self._incident_wave = self._incident_wave_initial.copy()
            self._aberrations_coefs = self._aberrations_coefs_initial.copy()
            self._image_shift_coefs = self._image_shift_coefs_initial.copy()
            self._chi_function = self._chi_function_initial.copy()
            self.error_iterations = []

            if self._object_optimizer == 'adam':
                self._object_grad_m = self._object_grad_m_initial.copy()
                self._object_grad_v = self._object_grad_v_initial.copy()

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

            self._active_tilt_index = 0

            object_sliced = self._project_sliced_object(
                self._object, self._num_slices
            )
            object_sliced_old = object_sliced.copy()

            # pad
            object_sliced = xp.pad(object_sliced, ((0,0) \
                                                    ,(self._object_padding_px[0],self._object_padding_px[1]) \
                                                    ,(self._object_padding_px[0],self._object_padding_px[1])))

            if a0 < fix_chi_func_iter:
                self._chi_function[self._active_tilt_index] =  (xp.matmul(self._aberrations_coefs[self._active_tilt_index], self._aberrations_basis.T) 
                                                            +xp.matmul(self._image_shift_coefs[self._active_tilt_index], self._image_shift_basis.T)
                                                            ).reshape(-1, self._padded_px[0], self._padded_px[1])

            # forward
            (
                propagated_waves,
                complex_object,
                predicted_detector_waves,
                self._residual_waves,
                error,
            ) = self._forward(
                object_sliced,
                self._incident_wave[self._active_tilt_index],
                self._amplitudes[self._active_tilt_index],
            )

            # self.predicted_exit_waves = predicted_exit_waves
            # self.exit_waves = self._predicted_exit_waves[self._active_tilt_index]

            # adjoint operator
            object_sliced, self._incident_wave[self._active_tilt_index] = self._adjoint(
                current_object=object_sliced,
                current_incident_wave=self._incident_wave[self._active_tilt_index],
                complex_object=complex_object,
                propagated_waves=propagated_waves,
                residual_waves=self._residual_waves,
                predicted_detector_waves=predicted_detector_waves,
                fix_chi_func= a0 < fix_chi_func_iter,
                fix_aberrations_coefs= a0 < fix_aberrations_coefs_iter,
                aberrations_coefs=self._aberrations_coefs[self._active_tilt_index],
                fix_image_shift_coefs= a0 < fix_image_shift_iter,
                image_shift_coefs=self._image_shift_coefs[self._active_tilt_index],
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
            error /= (
                self._object_shape[0]*self._object_shape[1]
                * self._num_defocus
            )
            
            self.error_iterations.append(error.item())
                    
            if store_iterations:
                self.object_iterations.append(asnumpy(self._object.copy()))
                self.incident_wave_iterations.append(asnumpy(self._incident_wave[:,self._cropping_px[0]:self._cropping_px[1],self._cropping_px[2]:self._cropping_px[3]].copy()))
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