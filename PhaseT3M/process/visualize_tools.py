# Juhyeok Lee, LBNL, 2024.
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


class Visualize_tools:

    def _crop_rotate_object_manually(
        self,
        array,
        angle,
        x_lims,
        y_lims,
    ):
        """
        Crops and rotates rotates object manually.

        Parameters
        ----------
        array: np.ndarray
            Object array to crop and rotate. Only operates on numpy arrays for comptatibility.
        angle: float
            In-plane angle in degrees to rotate by
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices

        Returns
        -------
        cropped_rotated_array: np.ndarray
            Cropped and rotated object array
        """

        asnumpy = self._asnumpy
        min_x, max_x = x_lims
        min_y, max_y = y_lims

        if angle is not None:
            rotated_array = rotate_np(
                asnumpy(array), angle, reshape=False, axes=(-2, -1)
            )
        else:
            rotated_array = asnumpy(array)

        return rotated_array[..., min_x:max_x, min_y:max_y]

    def _visualize_last_iteration(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_incident_wave: bool,
        plot_fourier_incident_wave: bool,
        plot_nth_incident_wave: int,
        plot_imaginary_object: bool,
        projection_angle_deg: float,
        projection_axes: Tuple[int, int],
        x_lims: Tuple[int, int],
        y_lims: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays last reconstructed object and incident wave iterations.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_incident_wave: bool
            If true, the reconstructed incident intensity is also displayed
        plot_fourier_incident_wave: bool, optional
            If true, the reconstructed complex Fourier incident wave is displayed
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        figsize = kwargs.pop("figsize", (8, 5))
        cmap = kwargs.pop("cmap", "magma")
        invert = kwargs.pop("invert", False)
        hue_start = kwargs.pop("hue_start", 0)

        asnumpy = self._asnumpy

        if projection_angle_deg is not None:
            rotated_3d_obj = self._rotate(
                self._object.real if plot_imaginary_object==False else self._object.imag,
                projection_angle_deg,
                axes=projection_axes,
                reshape=False,
                order=2,
            )
            rotated_3d_obj = asnumpy(rotated_3d_obj)
        else:
            rotated_3d_obj = self.object.real if plot_imaginary_object==False else self.object.imag
        
        #rotated_object = rotated_3d_obj.sum(0)
        rotated_object = self._crop_rotate_object_manually(
            rotated_3d_obj.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
        )
        rotated_shape = rotated_object.shape
        incident_wave_shape = self.incident_wave[0].shape

        object_extent = [
            0,
            self.sampling[1] * rotated_shape[1],
            self.sampling[0] * rotated_shape[0],
            0,
        ]
        
        if plot_fourier_incident_wave:
            incident_wave_extent = [
                -1/self.sampling[1] / 2,
                1/self.sampling[1] / 2,
                1/self.sampling[0] / 2,
                -1/self.sampling[0] / 2,
            ]
        elif plot_incident_wave:
            incident_wave_extent = [
                0,
                self.sampling[1] * incident_wave_shape[1],
                self.sampling[0] * incident_wave_shape[0],
                0,
            ]


        if plot_convergence:
            if plot_incident_wave or plot_fourier_incident_wave:
                spec = GridSpec(
                    ncols=2,
                    nrows=2,
                    height_ratios=[4, 1],
                    hspace=0.15,
                    #width_ratios=[
                    #    (extent[1] / extent[2]),
                    #    1,
                    #],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            if plot_incident_wave or plot_fourier_incident_wave:
                spec = GridSpec(
                    ncols=2,
                    nrows=1,
                    #width_ratios=[
                    #    (extent[1] / extent[2]),
                    #    1,
                    #],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=1, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)

        if plot_incident_wave or plot_fourier_incident_wave:
            # Object
            ax = fig.add_subplot(spec[0, 0])
            im = ax.imshow(
                rotated_object,
                extent=object_extent,
                cmap=cmap,
                **kwargs,
            )

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed object projection")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Initail wave
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)

            ax = fig.add_subplot(spec[0, 1])
            if plot_fourier_incident_wave:
                incident_wave_array = np.abs(np.fft.fftshift(np.fft.fft2(self.incident_wave[plot_nth_incident_wave])))
                #Complex2RGB(
                #    self.probe_fourier, hue_start=hue_start, invert=invert
                #)
                ax.set_title("F. inc. wave ({}/{})".format(plot_nth_incident_wave, len(self.incident_wave)-1))
                ax.set_ylabel("kx [mrad]")
                ax.set_xlabel("ky [mrad]")
            else:
                incident_wave_array = np.abs(self.incident_wave[plot_nth_incident_wave])
                #Complex2RGB(
                #    self.probe, hue_start=hue_start, invert=invert
                #)
                ax.set_title("R. inc. wave ({}/{})".format(plot_nth_incident_wave, len(self.incident_wave)-1))
                ax.set_ylabel("x [A]")
                ax.set_xlabel("y [A]")

            im = ax.imshow(
                incident_wave_array,
                extent=incident_wave_extent,
                **kwargs,
            )

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)
                #add_colorbar_arg(ax_cb, hue_start=hue_start, invert=invert)
        else:
            ax = fig.add_subplot(spec[0])
            im = ax.imshow(
                rotated_object,
                extent=incident_wave_extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed object projection")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "error_iterations"):
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            errors = np.array(self.error_iterations)
            if plot_incident_wave:
                ax = fig.add_subplot(spec[1, :])
            else:
                ax = fig.add_subplot(spec[1])
            ax.semilogy(np.arange(errors.shape[0]), errors, **kwargs)
            ax.set_ylabel("NMSE")
            ax.set_xlabel("Iteration Number")
            ax.yaxis.tick_right()

        fig.suptitle(f"Normalized Mean Squared Error: {self.error:.3e}")
        spec.tight_layout(fig)

    def _visualize_all_iterations(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_incident_wave: bool,
        plot_fourier_incident_wave: bool,
        plot_nth_incident_wave: int,
        plot_imaginary_object: bool,
        iterations_grid: Tuple[int, int],
        projection_angle_deg: float,
        projection_axes: Tuple[int, int],
        x_lims: Tuple[int, int],
        y_lims: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays all reconstructed object and probe iterations.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_incident_wave: bool
            If true, the reconstructed incident wave intensity is also displayed
        plot_fourier_incident_wave: bool, optional
            If true, the reconstructed complex Fourier incident wave is displayed
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
        """
        asnumpy = self._asnumpy

        if not hasattr(self, "object_iterations"):
            raise ValueError(
                (
                    "Object and incident wave iterations were not saved during reconstruction. "
                    "Please re-run using store_iterations=True."
                )
            )

        if iterations_grid == "auto":
            num_iter = len(self.error_iterations)

            if num_iter == 1:
                return self._visualize_last_iteration(
                    fig=fig,
                    plot_convergence=plot_convergence,
                    plot_incident_wave=plot_incident_wave,
                    plot_fourier_incident_wave=plot_fourier_incident_wave,
                    cbar=cbar,
                    projection_angle_deg=projection_angle_deg,
                    projection_axes=projection_axes,
                    x_lims=x_lims,
                    y_lims=y_lims,
                    **kwargs,
                )
            elif plot_incident_wave or plot_fourier_incident_wave:
                iterations_grid = (2, 4) if num_iter > 4 else (2, num_iter)
            else:
                iterations_grid = (2, 4) if num_iter > 8 else (2, num_iter // 2)
        else:
            if (plot_incident_wave or plot_fourier_incident_wave) and iterations_grid[0] != 2:
                raise ValueError()

        auto_figsize = (
            (3 * iterations_grid[1], 3 * iterations_grid[0] + 1)
            if plot_convergence
            else (3 * iterations_grid[1], 3 * iterations_grid[0])
        )
        figsize = kwargs.pop("figsize", auto_figsize)
        cmap = kwargs.pop("cmap", "magma")
        invert = kwargs.pop("invert", False)
        hue_start = kwargs.pop("hue_start", 0)

        errors = np.array(self.error_iterations)

        if projection_angle_deg is not None:
            objects = [
                self._crop_rotate_object_manually(
                    rotate_np(
                        obj.real if plot_imaginary_object==False else obj.imag,
                        projection_angle_deg,
                        axes=projection_axes,
                        reshape=False,
                        order=2,
                    ).sum(0),
                    angle=None,
                    x_lims=x_lims,
                    y_lims=y_lims,
                )
                for obj in self.object_iterations
            ]
        else:
            objects = [
                self._crop_rotate_object_manually(
                    obj.real.sum(0) if plot_imaginary_object==False else obj.imag.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
                )
                for obj in self.object_iterations
            ]

        if plot_incident_wave or plot_fourier_incident_wave:
            total_grids = (np.prod(iterations_grid) / 2).astype("int")
            #probes = self.probe_iterations
        else:
            total_grids = np.prod(iterations_grid)
        max_iter = len(objects) - 1
        grid_range = range(0, max_iter + 1, max_iter // (total_grids - 1))

        extent = [
            0,
            self.sampling[1] * objects[0].shape[1],
            self.sampling[0] * objects[0].shape[0],
            0,
        ]
        
        if plot_fourier_incident_wave:
            incident_wave_extent = [
                -1/self.sampling[1] / 2,
                1/self.sampling[1] / 2,
                1/self.sampling[0] / 2,
                -1/self.sampling[0] / 2,
            ]
        elif plot_incident_wave:
            incident_wave_extent = [
                0,
                self.sampling[1] * self.incident_wave[0].shape[1],
                self.sampling[0] * self.incident_wave[0].shape[0],
                0,
            ]

        if plot_convergence:
            if plot_incident_wave or plot_fourier_incident_wave:
                spec = GridSpec(ncols=1, nrows=3, height_ratios=[4, 4, 1], hspace=0)
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0)
        else:
            if plot_incident_wave or plot_fourier_incident_wave:
                spec = GridSpec(ncols=1, nrows=2)
            else:
                spec = GridSpec(ncols=1, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)
        
        grid = ImageGrid(
            fig,
            spec[0],
            nrows_ncols=(1, iterations_grid[1]) if plot_incident_wave else iterations_grid,
            axes_pad=(0.75, 0.5) if cbar else 0.5,
            cbar_mode="each" if cbar else None,
            cbar_pad="2.5%" if cbar else None,
        )
        
        for n, ax in enumerate(grid):
            im = ax.imshow(
                objects[grid_range[n]],
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_title(f"Iter: {grid_range[n]} Object")

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            if cbar:
                grid.cbar_axes[n].colorbar(im)
        
        if plot_incident_wave or plot_fourier_incident_wave:
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            grid = ImageGrid(
                fig,
                spec[1],
                nrows_ncols=(1, iterations_grid[1]),
                axes_pad=(0.75, 0.5) if cbar else 0.5,
                cbar_mode="each" if cbar else None,
                cbar_pad="2.5%" if cbar else None,
            )

            for n, ax in enumerate(grid):
                if plot_fourier_incident_wave:
                    incident_wave_array = np.abs(np.fft.fftshift(np.fft.fft2(self.incident_wave_iterations[grid_range[n]][plot_nth_incident_wave])))
                    #Complex2RGB(
                    #   asnumpy(
                    #        self._return_fourier_probe_from_centered_probe(
                    #            probes[grid_range[n]]
                    #        )
                    #    ),
                    #    hue_start=hue_start,
                    #    invert=invert,
                    #)
                    ax.set_title(f"Iter: {grid_range[n]} Fourier inc. wave ({plot_nth_incident_wave})")
                    ax.set_ylabel("kx [mrad]")
                    ax.set_xlabel("ky [mrad]")
                else:
                    incident_wave_array = np.abs(self.incident_wave_iterations[grid_range[n]][plot_nth_incident_wave])
                    #Complex2RGB(
                    #    probes[grid_range[n]], hue_start=hue_start, invert=invert
                    #)
                    ax.set_title(f"Iter: {grid_range[n]} inc. wave ({plot_nth_incident_wave})")
                    ax.set_ylabel("x [A]")
                    ax.set_xlabel("y [A]")

                im = ax.imshow(
                    incident_wave_array,
                    extent=incident_wave_extent,
                    **kwargs,
                )

                if cbar:
                    fig.add_axes(grid.cbar_axes[n])
                    fig.colorbar(im, cax=grid.cbar_axes[n])

        if plot_convergence:
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            if plot_incident_wave:
                ax2 = fig.add_subplot(spec[2])
            else:
                ax2 = fig.add_subplot(spec[1])
            ax2.semilogy(np.arange(errors.shape[0]), errors, **kwargs)
            ax2.set_ylabel("NMSE")
            ax2.set_xlabel("Iteration Number")
            ax2.yaxis.tick_right()

        spec.tight_layout(fig)
