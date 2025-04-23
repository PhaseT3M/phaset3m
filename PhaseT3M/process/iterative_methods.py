# Juhyeok Lee, LBNL, 2024.

import warnings, time
from typing import Mapping, Sequence, Tuple


from PhaseT3M.process.tqdmnd import tqdmnd
from PhaseT3M.datastack.datastack import DataStack

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
import os
from PhaseT3M.process.iterative_contraints import Contraints
from PhaseT3M.process.utils import (electron_wavelength_angstrom,
                              spatial_frequencies,
                              aberrations_basis_function,
                              gradient_strengh_correction,
                              fit_aberration_surface)

class Reconstruction_methods():

    def _precompute_propagator_arrays(
        self,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        energy: float,
        slice_thicknesses: Sequence[float],
    ):
        """
        Precomputes propagator arrays complex wave-function will be convolved by,
        for all slice thicknesses.

        Parameters
        ----------
        gpts: Tuple[int,int]
            Wavefunction pixel dimensions
        sampling: Tuple[float,float]
            Wavefunction sampling in A
        energy: float
            The electron energy of the wave functions in eV
        slice_thicknesses: Sequence[float]
            Array of slice thicknesses in A

        Returns
        -------
        propagator_arrays: np.ndarray
            (T,Sx,Sy) shape array storing propagator arrays
        """
        xp = self._xp

        # Frequencies
        kx, ky = spatial_frequencies(gpts, sampling)
        kx = xp.asarray(kx, dtype=xp.float32)
        ky = xp.asarray(ky, dtype=xp.float32)

        # Propagators
        wavelength = electron_wavelength_angstrom(energy)
        num_slices = slice_thicknesses.shape[0]
        propagators = xp.empty(
            (num_slices, kx.shape[0], ky.shape[0]), dtype=xp.complex64
        )
        for i, dz in enumerate(slice_thicknesses):
            propagators[i] = xp.exp(
                1.0j * (-(kx**2)[:, None] * np.pi * wavelength * dz)
            )
            propagators[i] *= xp.exp(
                1.0j * (-(ky**2)[None] * np.pi * wavelength * dz)
            )

        return propagators
    

    def _propagate_array(self, array: np.ndarray, propagator_array: np.ndarray):
        """
        Propagates array by Fourier convolving array with propagator_array.

        Parameters
        ----------
        array: np.ndarray
            Wavefunction array to be convolved
        propagator_array: np.ndarray
            Propagator array to convolve array with

        Returns
        -------
        propagated_array: np.ndarray
            Fourier-convolved array
        """
        xp = self._xp

        return xp.fft.ifft2(xp.fft.fft2(array) * propagator_array)
    

    def _project_sliced_object(self, array: np.ndarray, output_z):
        """
        Expands supersliced object or projects voxel-sliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to expand/project
        output_z: int
            Output_dimension to expand/project array to.
            If output_z > array.shape[0] array is expanded, else it's projected

        Returns
        -------
        expanded_or_projected_array: np.ndarray
            expanded or projected array
        """
        xp = self._xp
        input_z = array.shape[0]

        voxels_per_slice = np.ceil(input_z / output_z).astype("int")
        pad_size = voxels_per_slice * output_z - input_z

        padded_array = xp.pad(array, ((0, pad_size), (0, 0), (0, 0)))

        return xp.sum(
            padded_array.reshape(
                (
                    -1,
                    voxels_per_slice,
                )
                + array.shape[1:]
            ),
            axis=1,
        )

    def _expand_sliced_object(self, array: np.ndarray, output_z):
        """
        Expands supersliced object or projects voxel-sliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to expand/project
        output_z: int
            Output_dimension to expand/project array to.
            If output_z > array.shape[0] array is expanded, else it's projected

        Returns
        -------
        expanded_or_projected_array: np.ndarray
            expanded or projected array
        """
        xp = self._xp
        input_z = array.shape[0]

        voxels_per_slice = np.ceil(output_z / input_z).astype("int")
        remainder_size = voxels_per_slice - (voxels_per_slice * input_z - output_z)

        voxels_in_slice = xp.repeat(voxels_per_slice, input_z)
        voxels_in_slice[-1] = remainder_size if remainder_size > 0 else voxels_per_slice

        normalized_array = array / xp.asarray(voxels_in_slice)[:, None, None]
        return xp.repeat(normalized_array, voxels_per_slice, axis=0)[:output_z]
    
    def _normalize_intensities(self, intensities, amplitude_normalization = True):
        """
        Take square root and normalize

        Parameters
        ----------
        intensities: (defocus, Rx, Ry) np.ndarray
            Zero-padded  intensities

        Returns
        -------
        amplitudes: (defocus, Rx, Ry) np.ndarray
            Flat array of normalized diffraction amplitudes
        """
        xp = self._xp

        amplitudes = xp.zeros_like(intensities)
        intensities = self._asnumpy(intensities)
        amplitudes = self._asnumpy(amplitudes)

        for defocus_indx in range(intensities.shape[0]):
            intensities[defocus_indx] = np.maximum(intensities[defocus_indx], 0)
            if amplitude_normalization == True:
                normalize_factor = intensities.shape[1] * intensities.shape[2] / xp.sum(intensities[defocus_indx])
                amplitudes[defocus_indx] = np.sqrt(intensities[defocus_indx]* normalize_factor)
            else:
                amplitudes[defocus_indx] = np.sqrt(intensities[defocus_indx])

        amplitudes = xp.asarray(amplitudes, dtype=xp.float32)

        return amplitudes    



            
    def _projection(self, current_object, current_incident_wave):
        """
        Ptychographic overlap projection method.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_incident_wave: np.ndarray
            Current incident wave estimate

        Returns
        --------
        propagated_waves: np.ndarray
            Shifted waves at each layer
        complex_object: np.ndarray
            Patched object view
        predicted_exit_waves: np.ndarray
            exit waves after N propagations and N transmissions
        """

        xp = self._xp

        complex_object = xp.exp(1j * current_object)

        propagated_waves = xp.empty_like(complex_object)
        propagated_waves[0] = current_incident_wave
        

        for s in range(self._num_slices):
            # transmit
            transmitted_waves = complex_object[s] * propagated_waves[s]

            # propagate
            if s + 1 < self._num_slices:
                propagated_waves[s + 1] = self._propagate_array(
                    transmitted_waves, self._propagator_arrays[s]
                )
        
        predicted_exit_waves = xp.expand_dims(transmitted_waves, axis=0)

        return propagated_waves, complex_object, predicted_exit_waves
    

    def _gradient_descent_real_projection(self, amplitudes, predicted_exit_waves):
        """
        Real projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        predicted_exit_waves: np.ndarray
            Predicted exit waves after N propagations and N transmissions

        Returns
        --------
        residual_waves: np.ndarray
            Updated exit wave difference
        error: float
            Reconstruction error
        """

        xp = self._xp

        # Crop
        residual_waves = predicted_exit_waves[:,self._cropping_px[0]:self._cropping_px[1],self._cropping_px[2]:self._cropping_px[3]]
        
        error = xp.sum(xp.abs(amplitudes - xp.abs(residual_waves)) ** 2)

        modified_exit_wave = amplitudes * xp.exp(1j * xp.angle(residual_waves))

        residual_waves = modified_exit_wave - residual_waves
        
        # Pad
        residual_waves = xp.pad(residual_waves, ((0,0) \
                                        ,(self._object_padding_px[0], self._object_padding_px[1]) \
                                        ,(self._object_padding_px[0], self._object_padding_px[1])))

        return residual_waves, error
    

    def _projection_sets_real_projection(
        self,
        amplitudes,
        predicted_exit_waves,
        residual_waves,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Real projection method for DM_AP and RAAR methods.
        Generalized projection using three parameters: a,b,c

            DM_AP(\\alpha)   :   a =  -\\alpha, b = 1, c = 1 + \\alpha
              DM: DM_AP(1.0), AP: DM_AP(0.0)

            RAAR(\\beta)     :   a = 1-2\\beta, b = \beta, c = 2
              DM : RAAR(1.0)

            RRR(\\gamma)     :   a = -\\gamma, b = \\gamma, c = 2
              DM: RRR(1.0)

            SUPERFLIP       :   a = 0, b = 1, c = 2

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        transmitted_waves: np.ndarray
            Transmitted waves after N-1 propagations and N transmissions
        predicted_exit_waves: np.ndarray
            previously estimated exit waves
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        residual_waves: np.ndarray
            Updated exit wave difference
        error: float
            Reconstruction error
        """

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        # Crop
        predicted_exit_waves = predicted_exit_waves[:,self._cropping_px[0]:self._cropping_px[1],self._cropping_px[2]:self._cropping_px[3]]
        
        if residual_waves is None:
            residual_waves = predicted_exit_waves.copy()
        
        error = xp.sum(xp.abs(amplitudes - xp.abs(predicted_exit_waves)) ** 2)

        factor_to_be_projected = (
            projection_c * predicted_exit_waves + projection_y * residual_waves
        )
        
        real_projected_factor = amplitudes * xp.exp(
            1j * xp.angle(factor_to_be_projected)
        )
        
        residual_waves = (
            projection_x * residual_waves
            + projection_a * predicted_exit_waves
            + projection_b * real_projected_factor
        )

        # Pad
        residual_waves = xp.pad(residual_waves, ((0,0) \
                                        ,(self._object_padding_px[0], self._object_padding_px[1]) \
                                        ,(self._object_padding_px[0], self._object_padding_px[1])))

        return residual_waves, error

    def _forward(
        self,
        current_object,
        current_incident_wave,
        amplitudes,
        exit_waves,
        use_projection_scheme,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Forward operator.
        Calls _projection() and the appropriate _fourier_projection().

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_incident_wave: np.ndarray
            Current incident wave estimate
        amplitudes: np.ndarray
            Normalized measured amplitudes
        exit_waves: np.ndarray
            previously estimated exit waves
        use_projection_scheme: bool,
            If True, use generalized projection update
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        propagated_waves:np.ndarray
            waves[object^n]
        complex_object: np.ndarray
            complex object 
        predicted_exit_waves: np.ndarray
            exit wave right before sample
        residual_waves: np.ndarray
            Updated exit wave difference
        error: float
            Reconstruction error
        """
        xp = self._xp

        # Beam propagation
        (
            propagated_waves,
            complex_object,
            predicted_exit_waves,
        ) = self._projection(current_object, current_incident_wave)
        
        self._predicted_exit_waves[self._active_tilt_index] = xp.squeeze(predicted_exit_waves).copy()

        # apply transfer function
        self.transfer_functions = (xp.exp( -1.0j * self._chi_function[self._active_tilt_index]
                    ).reshape(-1, self._padded_px[0], self._padded_px[1])*self._temperal_coherence_envelop_function).astype(xp.complex64)

        predicted_exit_waves = self._propagate_array(
                predicted_exit_waves, self.transfer_functions
                )

        
        if use_projection_scheme:
            (
                exit_waves[self._active_tilt_index],
                error,
            ) = self._projection_sets_real_projection(
                amplitudes,
                predicted_exit_waves,
                exit_waves[self._active_tilt_index],
                projection_a,
                projection_b,
                projection_c,
            )
        else:
            residual_waves, error = self._gradient_descent_real_projection(
                amplitudes, predicted_exit_waves
            )

        return propagated_waves, complex_object, predicted_exit_waves, residual_waves, error

    def _imageshift_gradient_descent_adjoint(
        self,
        image_shift_coefs,
        predicted_exit_waves,
        residual_waves,
        aberrations_basis,
        step_size,
        normalization_min,
    ):
        """
        Gradient for aberrations correction
        Computes aberration coefs gradient and update.

        Parameters
        --------
        image_shift_coefs: np.ndarray
            Current image shift coefficients
        predicted_exit_waves: np.ndarray
            predicted exit waves estimate
        residual_waves: np.ndarray
            Updated residual_waves difference
        aberrations_basis: np.ndarray
            aberrations basis
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity

        Returns
        --------
        aberrations_coefs: np.ndarray
            Updated aberrations coefficients
        """

        xp = self._xp
        beta = [0.9, 0.999]
        eps = 1e-24

        # aberration coefs-update
        predicted_exit_waves = xp.conj(xp.fft.fft2(predicted_exit_waves)) 
        # exit_normalization = xp.abs(predicted_exit_waves) ** 2 
        # exit_normalization = 1 / xp.sqrt(
        #     1e-16
        #     + ((1 - normalization_min) * exit_normalization) ** 2
        #     + (normalization_min * xp.max(exit_normalization)) ** 2
        # )

        tmp_residual = (predicted_exit_waves*xp.fft.fft2(residual_waves)).reshape(-1, self._padded_px[0]*self._padded_px[1])
        aberrations_grad = xp.real(2j*xp.matmul(tmp_residual, aberrations_basis)) 

        # adam
        self._image_shift_coefs_m[self._active_tilt_index] = beta[0]*self._image_shift_coefs_m[self._active_tilt_index] + (1-beta[0])*aberrations_grad
        self._image_shift_coefs_v[self._active_tilt_index] = beta[1]*self._image_shift_coefs_v[self._active_tilt_index] + (1-beta[1])*aberrations_grad**2
        

        image_shift_coefs += step_size * (self._image_shift_coefs_m[self._active_tilt_index]/(1-beta[0]**(self._active_iter+1))) \
                                    /(xp.sqrt(self._image_shift_coefs_v[self._active_tilt_index]/(1-beta[1]**(self._active_iter+1)))+eps)       

        return image_shift_coefs

    def _aberrations_gradient_descent_adjoint(
        self,
        aberrations_coefs,
        predicted_exit_waves,
        residual_waves,
        aberrations_basis,
        step_size,
        normalization_min,
    ):
        """
        Gradient for aberrations correction
        Computes aberration coefs gradient and update.

        Parameters
        --------
        aberrations_coefs: np.ndarray
            Current aberrations coefficients
        predicted_exit_waves: np.ndarray
            predicted exit waves estimate
        residual_waves: np.ndarray
            Updated residual_waves difference
        aberrations_basis: np.ndarray
            aberrations basis
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity

        Returns
        --------
        aberrations_coefs: np.ndarray
            Updated aberrations coefficients
        """

        xp = self._xp
        beta = [0.9, 0.999]
        eps = 1e-24

        # aberration coefs-update
        predicted_exit_waves = xp.conj(xp.fft.fft2(predicted_exit_waves)) 
        # exit_normalization = xp.abs(predicted_exit_waves) ** 2 
        # exit_normalization = 1 / xp.sqrt(
        #     1e-16
        #     + ((1 - normalization_min) * exit_normalization) ** 2
        #     + (normalization_min * xp.max(exit_normalization)) ** 2
        # )

        tmp_residual = (predicted_exit_waves*xp.fft.fft2(residual_waves)).reshape(-1, self._padded_px[0]*self._padded_px[1])
        aberrations_grad = xp.real(2j*xp.matmul(tmp_residual, aberrations_basis)) 

        # adam
        self._aberrations_coefs_m[self._active_tilt_index] = beta[0]*self._aberrations_coefs_m[self._active_tilt_index] + (1-beta[0])*aberrations_grad
        self._aberrations_coefs_v[self._active_tilt_index] = beta[1]*self._aberrations_coefs_v[self._active_tilt_index] + (1-beta[1])*aberrations_grad**2
        

        aberrations_coefs += step_size * (self._aberrations_coefs_m[self._active_tilt_index]/(1-beta[0]**(self._active_iter+1))) \
                                    /(xp.sqrt(self._aberrations_coefs_v[self._active_tilt_index]/(1-beta[1]**(self._active_iter+1)))+eps)       

        return aberrations_coefs
 
    # def _aberrations_gradient_descent_adjoint(
    #     self,
    #     aberrations_coefs,
    #     predicted_exit_waves,
    #     residual_waves,
    #     aberrations_basis,
    #     step_size,
    #     normalization_min,
    # ):
    #     """
    #     Gradient for aberrations correction
    #     Computes aberration coefs gradient and update.

    #     Parameters
    #     --------
    #     aberrations_coefs: np.ndarray
    #         Current aberrations coefficients
    #     predicted_exit_waves: np.ndarray
    #         predicted exit waves estimate
    #     residual_waves: np.ndarray
    #         Updated residual_waves difference
    #     aberrations_basis: np.ndarray
    #         aberrations basis
    #     step_size: float, optional
    #         Update step size
    #     normalization_min: float, optional
    #         Probe normalization minimum as a fraction of the maximum overlap intensity

    #     Returns
    #     --------
    #     aberrations_coefs: np.ndarray
    #         Updated aberrations coefficients
    #     """

    #     xp = self._xp

    #     # aberration coefs-update
    #     predicted_exit_waves = xp.conj(xp.fft.fft2(predicted_exit_waves)) 
    #     exit_normalization = xp.abs(predicted_exit_waves) ** 2 
    #     exit_normalization = 1 / xp.sqrt(
    #         1e-16
    #         + ((1 - normalization_min) * exit_normalization) ** 2
    #         + (normalization_min * xp.max(exit_normalization)) ** 2
    #     )
        
    #     tmp_residual = predicted_exit_waves*xp.fft.fft2(residual_waves)*exit_normalization
    #     #tmp_residual = xp.fft.fft2(xp.conj(predicted_exit_waves))*xp.fft.fft2(residual_waves)*exit_normalization
    #     tmp_residual = tmp_residual.reshape(-1, self._padded_px[0]*self._padded_px[1])

    #     #aberrations_coefs += step_size * xp.real(2j*xp.matmul(tmp_residual, aberrations_basis))        
    #     aberrations_coefs += step_size * 1e5 * xp.real(2j*xp.matmul(tmp_residual, aberrations_basis/xp.max(aberrations_basis, axis=0)))  # 1e5 -> adjust step size

    #     return aberrations_coefs


    def _chi_gradient_descent_adjoint(
        self,
        chi_function,
        predicted_exit_waves,
        residual_waves,
        step_size,
        normalization_min,
    ):
        """
        Gradient for chi function correction
        Computes chi function gradient and update.

        Parameters
        --------
        chi_function: np.ndarray
            chi function
        predicted_exit_waves: np.ndarray
            predicted exit waves estimate
        residual_waves: np.ndarray
            Updated residual_waves difference
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity

        Returns
        --------
        chi_function: np.ndarray
            Updated chi function
        """

        xp = self._xp
        beta = [0.9, 0.999]
        eps = 1e-4

        # aberration coefs-update
        predicted_exit_waves = xp.conj(xp.fft.fft2(predicted_exit_waves)) 
        exit_normalization = xp.abs(predicted_exit_waves) ** 2 
        exit_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * exit_normalization) ** 2
            + (normalization_min * xp.max(exit_normalization)) ** 2
        )
        
        tmp_residual = predicted_exit_waves*xp.fft.fft2(residual_waves)*exit_normalization
        chi_function_grad = xp.real(2j*tmp_residual)

        # adam
        self._chi_function_m[self._active_tilt_index] = beta[0]*self._chi_function_m[self._active_tilt_index] + (1-beta[0])*chi_function_grad
        self._chi_function_v[self._active_tilt_index] = beta[1]*self._chi_function_v[self._active_tilt_index] + (1-beta[1])*chi_function_grad**2
        

        chi_function += step_size * (self._chi_function_m[self._active_tilt_index]/(1-beta[0]**(self._active_iter+1))) \
                                    /(xp.max(xp.sqrt(self._chi_function_v[self._active_tilt_index]/(1-beta[1]**(self._active_iter+1))))+eps)       


        return chi_function
    

    # def _chi_gradient_descent_adjoint(
    #     self,
    #     chi_function,
    #     predicted_exit_waves,
    #     residual_waves,
    #     step_size,
    #     normalization_min,
    # ):
    #     """
    #     Gradient for chi function correction
    #     Computes chi function gradient and update.

    #     Parameters
    #     --------
    #     chi_function: np.ndarray
    #         chi function
    #     predicted_exit_waves: np.ndarray
    #         predicted exit waves estimate
    #     residual_waves: np.ndarray
    #         Updated residual_waves difference
    #     step_size: float, optional
    #         Update step size
    #     normalization_min: float, optional
    #         Probe normalization minimum as a fraction of the maximum overlap intensity

    #     Returns
    #     --------
    #     chi_function: np.ndarray
    #         Updated chi function
    #     """

    #     xp = self._xp

    #     # aberration coefs-update
    #     predicted_exit_waves = xp.conj(xp.fft.fft2(predicted_exit_waves)) 
    #     exit_normalization = xp.abs(predicted_exit_waves) ** 2 
    #     exit_normalization = 1 / xp.sqrt(
    #         1e-16
    #         + ((1 - normalization_min) * exit_normalization) ** 2
    #         + (normalization_min * xp.max(exit_normalization)) ** 2
    #     )
        
    #     tmp_residual = predicted_exit_waves*xp.fft.fft2(residual_waves)*exit_normalization
    #     #tmp_residual = xp.fft.fft2(xp.conj(predicted_exit_waves))*xp.fft.fft2(residual_waves)*exit_normalization
      
    #     chi_function += step_size * xp.real(2j*tmp_residual)

    #     return chi_function
    

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_incident_wave,
        complex_object,
        propagated_waves,
        residual_waves,
        step_size,
        normalization_min,
        fix_incident_wave
    ):
        """
        Adjoint operator for GD method.
        Computes object and incident wave update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_incident_wave: np.ndarray
            Current incident wave estimate
        complex_object: np.ndarray
            Complex object view
        propagated_waves: np.ndarray
            Shifted waves at each layer
        residual_waves: np.ndarray
            Updated exit_waves difference
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_incident_wave: bool, optional
            If True, incident wave will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_incident_wave: np.ndarray 
            Updated incident_wave estimate
        """
        xp = self._xp
        
        for s in reversed(range(self._num_slices)):

            save_waves = propagated_waves[s]
            obj = complex_object[s]          
            
            # object-update
            incident_normalization = xp.abs(save_waves) ** 2
            incident_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * incident_normalization) ** 2
                + (normalization_min * xp.max(incident_normalization)) ** 2
            )

            if self._object_type == "complex":
                current_object[s] += step_size * (
                        (-1j * xp.conj(obj) * xp.conj(save_waves) * residual_waves)
                    * incident_normalization
                )
            else:
                current_object[s] += step_size * (
                        xp.real(-1j * xp.conj(obj) * xp.conj(save_waves) * residual_waves)
                    * incident_normalization
                )

                # current_object[s] += step_size * gradient_strengh_correction(xp.real(-1j * xp.conj(obj) * xp.conj(save_waves) * residual_waves),
                #                                                                      self._pixel_sizes, energy=self._energy, xp=self._xp)


            # back-transmit
            residual_waves *= xp.conj(obj)

            if s > 0:
                # back-propagate
                residual_waves = self._propagate_array(
                        residual_waves, xp.conj(self._propagator_arrays[s-1])
                    )
                
            elif not fix_incident_wave:
                # incident wave update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )                

                current_incident_wave += (
                step_size
                * residual_waves
                * object_normalization
                )            

        return current_object, current_incident_wave

    def _projection_sets_adjoint(
        self,
        current_object,
        current_incident_wave,
        complex_object,
        propagated_waves,
        residual_waves,
        step_size,
        normalization_min,
        fix_incident_wave,
    ):
        """
        Adjoint operator for GD method.
        Computes object and incident wave update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_incident_wave: np.ndarray
            Current incident wave estimate
        complex_object: np.ndarray
            Complex object view
        propagated_waves: np.ndarray
            Shifted waves at each layer
        residual_waves: np.ndarray
            Updated exit_waves difference
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_incident_wave: bool, optional
            If True, incident wave will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_incident_wave: np.ndarray 
            Updated incident_wave estimate
        """
        xp = self._xp

        normalization_factor = xp.abs(current_incident_wave) ** 2
        normalization_factor = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * normalization_factor) ** 2
            + (normalization_min * xp.max(normalization_factor)) ** 2
        )

        # careful not to modify exit_waves in-place for projection set methods
        residual_waves_copy = residual_waves.copy()
        for s in reversed(range(self._num_slices)):
            save_waves = propagated_waves[s]
            obj = complex_object[s]          

            # object-update
            incident_normalization = xp.abs(save_waves) ** 2
            incident_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * incident_normalization) ** 2
                + (normalization_min * xp.max(incident_normalization)) ** 2
            )

            if self._object_type == "complex":
                current_object[s] += step_size * (
                        (-1j * xp.conj(obj) * xp.conj(save_waves) * residual_waves_copy)
                    * incident_normalization
                )
            else:
                current_object[s] += step_size * (
                        xp.real(-1j * xp.conj(obj) * xp.conj(save_waves) * residual_waves_copy)
                    * incident_normalization
                )

            # back-transmit
            residual_waves_copy *= xp.conj(obj)

            if s > 0:
                # back-propagate
                residual_waves_copy = self._propagate_array(
                        residual_waves_copy, xp.conj(self._propagator_arrays[s-1])
                    )
            elif not fix_incident_wave:
                # incident wave update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )                

                current_incident_wave += (
                step_size
                * residual_waves_copy
                * object_normalization
                )       


        return current_object, current_incident_wave


    def _adjoint(
        self,
        current_object,
        current_incident_wave,
        complex_object,
        propagated_waves,
        residual_waves,
        fix_chi_func: bool,
        fix_aberrations_coefs: bool,
        aberrations_coefs,
        fix_image_shift_coefs: bool,
        image_shift_coefs,
        predicted_exit_waves,
        use_projection_scheme: bool,
        step_size: float,
        chi_func_step_size: float,
        aberrations_step_size: float,
        image_shift_step_size: float,
        normalization_min: float,
        fix_incident_wave: bool,
    ):
        """
        Adjoint operator for GD method.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_incident_wave: np.ndarray
            Current incident wave estimate
        complex_object: np.ndarray
            Patched object view
        transmitted_probes: np.ndarray
            Transmitted probes at each layer
        residual_waves: np.ndarray
            Updated exit_waves difference
        fix_aberrations_coefs
            If True, don't use aberrations correction using gradient method
        aberrations_coefs: np.ndarray
            Current aberrations coefficients
        predicted_exit_waves: np.ndarray
            predicted exit waves estimate
        use_projection_scheme: bool,
            If True, use generalized projection update
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_incident_wave: bool, optional
            If True, incident_wave will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_incident_wave: np.ndarray
            Updated incident_wave estimate
        """
        xp = self._xp
            
        # image shift or aberrations correction
        if not fix_chi_func:
            # chi function correction
            self._chi_function[self._active_tilt_index] = self._chi_gradient_descent_adjoint(
                self._chi_function[self._active_tilt_index],
                predicted_exit_waves,
                residual_waves,
                chi_func_step_size,
                normalization_min,
            )    
        else:
            # image shift correction
            if not fix_image_shift_coefs:
                image_shift_coefs= self._imageshift_gradient_descent_adjoint(
                    image_shift_coefs,
                    predicted_exit_waves,
                    residual_waves,
                    self._image_shift_basis,
                    image_shift_step_size,
                    normalization_min,
                )
            # aberrations correction
            if not fix_aberrations_coefs:
                aberrations_coefs = self._aberrations_gradient_descent_adjoint(
                    aberrations_coefs,
                    predicted_exit_waves,
                    residual_waves,
                    self._aberrations_basis,
                    aberrations_step_size,
                    normalization_min,
                )

        # apply conjugate transfer function  
        residual_waves = self._propagate_array(
                residual_waves, xp.conj(self.transfer_functions)
                )

        residual_waves = xp.mean(residual_waves, axis=0)
        
        # Backpropagation
        if use_projection_scheme:
            current_object, current_incident_wave = self._projection_sets_adjoint(
                current_object,
                current_incident_wave,
                complex_object,
                propagated_waves,
                residual_waves[self._active_tilt_index],
                normalization_min,
                fix_incident_wave,
            )
        else:
            current_object, current_incident_wave = self._gradient_descent_adjoint(
                current_object,
                current_incident_wave,
                complex_object,
                propagated_waves,
                residual_waves,
                step_size,
                normalization_min,
                fix_incident_wave,
            )             

        return current_object, current_incident_wave


    

