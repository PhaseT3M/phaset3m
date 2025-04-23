# Juhyeok Lee, LBNL, 2024.
# This code is for 3D rotation

import warnings
import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
    
# def generate_grid_1d(shape, pixel_size=1, flag_fourier=False, xp=np):
#     """
#     Generates a 1D grid, centered at the middle of the array.
#     """
#     pixel_size = 1.0 / pixel_size / shape if flag_fourier else pixel_size
#     x_lin = (xp.arange(shape, dtype=xp.float32) - shape//2) * pixel_size
#     if flag_fourier:
#         x_lin = xp.fft.ifftshift(x_lin)#xp.roll(x_lin, - (shape-1) // 2)
#         # if shape %2 ==0:
#         #     x_lin[shape//2] = 0
#     return x_lin

def generate_grid_1d(shape, pixel_size=1, flag_fourier=False, xp=np):
    """
    Generates a 1D grid, centered at the middle of the array.
    """
    if flag_fourier:
        x_lin = (xp.arange(shape, dtype=xp.float32) - (shape) // 2) / pixel_size / shape
        x_lin = xp.fft.ifftshift(x_lin)

    else:
        x_lin = (xp.arange(shape, dtype=xp.float32) - (shape -1)/ 2) * pixel_size
        #x_lin = (xp.arange(shape, dtype=xp.float32) - (shape)// 2) * pixel_size

    return x_lin

class Image3DRotation:
    """
    A rotation class to compute 3D rotation using FFT shearing method or real space interpolation(cubic).
    """
    def __init__(self, shape, rot_method = "interp", object_type = "potential", xp = np, MEMORY_MAX_DIM= 600*600*600):
        self._rot_method = rot_method
        self._xp = xp
        self._object_type = object_type
        self._dim = np.array(shape)
        if rot_method == "Fourier_sheer":
            self.nx = generate_grid_1d(self._dim[0], xp=self._xp).reshape(-1, 1, 1)
            self.ny = generate_grid_1d(self._dim[1], xp=self._xp).reshape(1, -1, 1)
            self.nz = generate_grid_1d(self._dim[2], xp=self._xp).reshape(1, 1, -1)

            self.kx = generate_grid_1d(self._dim[0], flag_fourier=True, xp=self._xp).reshape(-1, 1, 1)
            self.ky = generate_grid_1d(self._dim[1], flag_fourier=True, xp=self._xp).reshape(1, -1, 1)
            self.kz = generate_grid_1d(self._dim[2], flag_fourier=True, xp=self._xp).reshape(1, 1, -1)

            self._swap_to_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

            # Compute FFTs sequentially if object size is too large
            self._slice_per_tile = max(int(np.min([np.floor(MEMORY_MAX_DIM * np.max(self._dim) / np.prod(self._dim)), np.max(self._dim)])), 1)

        else:
            if xp == cp:
                from cupyx.scipy.ndimage import affine_transform
            else:
                from scipy.ndimage import affine_transform

            self._affine_transform = affine_transform
            self._swap_zxy_to_xyz = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def _obtain_quaternion_from_matrix(self, rotmat):
        Tr = rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2]
        if Tr > 0:
            S = np.sqrt(Tr + 1.0) * 2  # S=4*qw
            qw = 0.25 * S
            qx = (rotmat[2, 1] - rotmat[1, 2]) / S
            qy = (rotmat[0, 2] - rotmat[2, 0]) / S
            qz = (rotmat[1, 0] - rotmat[0, 1]) / S
        elif (rotmat[0, 0] > rotmat[1, 1]) and (rotmat[0, 0] > rotmat[2, 2]):
            S = np.sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2]) * 2  # S=4*qx
            qw = (rotmat[2, 1] - rotmat[1, 2]) / S
            qx = 0.25 * S
            qy = (rotmat[0, 1] + rotmat[1, 0]) / S
            qz = (rotmat[0, 2] + rotmat[2, 0]) / S
        elif rotmat[1, 1] > rotmat[2, 2]:
            S = np.sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2]) * 2  # S=4*qy
            qw = (rotmat[0, 2] - rotmat[2, 0]) / S
            qx = (rotmat[0, 1] + rotmat[1, 0]) / S
            qy = 0.25 * S
            qz = (rotmat[1, 2] + rotmat[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1]) * 2  # S=4*qz
            qw = (rotmat[1, 0] - rotmat[0, 1]) / S
            qx = (rotmat[0, 2] + rotmat[2, 0]) / S
            qy = (rotmat[1, 2] + rotmat[2, 1]) / S
            qz = 0.25 * S

        return qx, qy, qz, qw


    def _calculate_real_sheer_coeffs(self, X_in, Y_in, Z_in, W, numPerm, Backflag=False):
        PermXYZ = np.roll([X_in, Y_in, Z_in], -1*numPerm)
        X = PermXYZ[0]
        Y = PermXYZ[1]
        Z = PermXYZ[2]
        if Backflag == True:
            W *= -1

        Singular = False
        Coeffs = np.zeros(8, dtype=np.float32)
        if abs(Y*Z - X*W) > 1e-5 and abs(X*Y - Z*W) > 1e-5:
            Coeffs[0] = (X**2 + Y**2) / (Y*Z - X*W)
            Coeffs[1] = (Y**2 - Z**2) / (X*Y - Z*W) - 2 * (Y*Z * (X**2 + Y**2)) / (X*Y - Z*W) / (Y*Z - X*W)
            Coeffs[2] = 2 * Y * Z / (X*Y - Z*W)
            Coeffs[3] = 2 * (X*W - Y*Z)
            Coeffs[4] = 2 * (X*Y - Z*W)
            Coeffs[5] = -2 * X * Y / (Y*Z - X*W)
            Coeffs[6] = (X**2 - Y**2) / (Y*Z - X*W) + 2 * X * Y * (Y**2 + Z**2) / (X*Y - Z*W) / (Y*Z - X*W)
            Coeffs[7] = (-1 + X**2 + W**2) / (X*Y - Z*W)
        elif abs(Y) < 1e-5:
            Coeffs[0] = -X / W
            Coeffs[1] = Z / W
            Coeffs[2] = 0
            Coeffs[3] = 2 * X * W
            Coeffs[4] = -2 * Z * W
            Coeffs[5] = 0
            Coeffs[6] = -X / W
            Coeffs[7] = Z / W
        else:
            Singular = True

        if Singular:
            absSum = float('inf')
        else:
            absSum = np.max(np.abs(Coeffs))
            #absSum = np.sum(np.abs(Coeffs))

        if Backflag == True:
            Coeffs *= -1
            
        return Coeffs, absSum
    

    def _shear_x_for_3D_rot(self, vol, a, b):
        xp = self._xp

        for idx_start in range(0, self._dim[0], self._slice_per_tile):
            idx_end = min(self._dim[0], idx_start + self._slice_per_tile)
            idx_slice = slice(idx_start, idx_end)

            if np.abs(b) > 1e-5:
                vol[idx_slice,:,:] = xp.fft.fft(vol[idx_slice,:,:], axis=2)
            if np.abs(a) > 1e-5:
                vol[idx_slice,:,:] = xp.fft.fft(vol[idx_slice,:,:], axis=1)
            if (np.abs(a) > 1e-5) or (np.abs(b) > 1e-5):
                vol[idx_slice,:,:] *= xp.exp(1j * 2 * xp.pi * (self._dim[1] / self._dim[0] * a * self.ky + self._dim[2] / self._dim[0] * b * self.kz) * self.nx[idx_slice,:,:])
            if np.abs(a) > 1e-5:
                vol[idx_slice,:,:] = xp.fft.ifft(vol[idx_slice,:,:], axis=1)
            if np.abs(b) > 1e-5:    
                vol[idx_slice,:,:] = xp.fft.ifft(vol[idx_slice,:,:], axis=2)
        
        return vol

    def _shear_y_for_3D_rot(self, vol, a, b):
        xp = self._xp

        for idx_start in range(0, self._dim[1], self._slice_per_tile):
            idx_end = min(self._dim[1], idx_start + self._slice_per_tile)
            idx_slice = slice(idx_start, idx_end)

            if np.abs(a) > 1e-5:
                vol[:,idx_slice,:] = xp.fft.fft(vol[:,idx_slice,:], axis=2)
            if np.abs(b) > 1e-5:
                vol[:,idx_slice,:] = xp.fft.fft(vol[:,idx_slice,:], axis=0)
            if (np.abs(a) > 1e-5) or (np.abs(b) > 1e-5):
                vol[:,idx_slice,:] *= xp.exp(1j * 2 * xp.pi * (self._dim[2] / self._dim[1] * a * self.kz + self._dim[0] / self._dim[1] * b * self.kx) * self.ny[:,idx_slice,:])
            if np.abs(b) > 1e-5:
                vol[:,idx_slice,:] = xp.fft.ifft(vol[:,idx_slice,:], axis=0)
            if np.abs(a) > 1e-5:    
                vol[:,idx_slice,:] = xp.fft.ifft(vol[:,idx_slice,:], axis=2)
        
        return vol

    def _shear_z_for_3D_rot(self, vol, a, b):
        xp = self._xp

        for idx_start in range(0, self._dim[2], self._slice_per_tile):
            idx_end = min(self._dim[2], idx_start + self._slice_per_tile)
            idx_slice = slice(idx_start, idx_end)

            if np.abs(b) > 1e-5:
                vol[:,:,idx_slice] = xp.fft.fft(vol[:,:,idx_slice], axis=1)
            if np.abs(a) > 1e-5:
                vol[:,:,idx_slice] = xp.fft.fft(vol[:,:,idx_slice], axis=0)
            if (np.abs(a) > 1e-5) or (np.abs(b) > 1e-5):
                vol[:,:,idx_slice] *= xp.exp(1j * 2 * xp.pi * (self._dim[0] / self._dim[2] * a * self.kx + self._dim[1] / self._dim[2] * b * self.ky) * self.nz[:,:,idx_slice])
            if np.abs(a) > 1e-5:
                vol[:,:,idx_slice] = xp.fft.ifft(vol[:,:,idx_slice], axis=0)
            if np.abs(b) > 1e-5:
                vol[:,:,idx_slice] = xp.fft.ifft(vol[:,:,idx_slice], axis=1)
        
        return vol

    def _shear_transforms_for_3D_rot(self, vol, a, b, dim):
        if dim == 0:
            vol = self._shear_x_for_3D_rot(vol, a, b)
        elif dim == 1:
            vol = self._shear_y_for_3D_rot(vol, a, b)
        elif dim == 2:
            vol = self._shear_z_for_3D_rot(vol, a, b)
        return vol


    def _Fourier_sheer_rotate_3d(
        self,
        obj,
        rotmat
    ):
        """ """

        # convert rotation matrix to quaterion
        rotmat = np.asarray(self._swap_to_xyz.T @ rotmat @ self._swap_to_xyz)
        qx, qy, qz, qw = self._obtain_quaternion_from_matrix(rotmat)

        Ang = np.degrees(np.arccos(qw)) * 2 

        Vec = np.array([qx, qy, qz])
        if np.linalg.norm(Vec) != 0:
            Vec = Vec / np.linalg.norm(Vec)

        rotStepInit = 20
        numStep = int(np.floor(Ang / rotStepInit) + 1)
        rotStep = Ang / numStep

        # Rotation using the input rotStep
        new_qw = np.cos(np.radians(rotStep) / 2)
        new_qx = np.sin(np.radians(rotStep) / 2) * Vec[0]
        new_qy = np.sin(np.radians(rotStep) / 2) * Vec[1]
        new_qz = np.sin(np.radians(rotStep) / 2) * Vec[2]

        # calculate all possible sheer cofficients
        sheer_coeffs_list = np.zeros([6,8])
        absSum_list = np.zeros(6)

        for i in range(3):
            sheer_coeffs_list[i], absSum_list[i] = self._calculate_real_sheer_coeffs(new_qx, new_qy, new_qz, new_qw, i, Backflag=False)
            sheer_coeffs_list[i+3], absSum_list[i+3] = self._calculate_real_sheer_coeffs(new_qx, new_qy, new_qz, new_qw, i, Backflag=True)

        min_inx = np.argmin(absSum_list)
        
        rot_obj = obj
        for i in range(numStep):
            if min_inx < 3:
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][6], sheer_coeffs_list[min_inx][7], (min_inx+0)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][4], sheer_coeffs_list[min_inx][5], (min_inx+2)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][2], sheer_coeffs_list[min_inx][3], (min_inx+1)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][0], sheer_coeffs_list[min_inx][1], (min_inx+0)%3)
            else:
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][0], sheer_coeffs_list[min_inx][1], (min_inx+0)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][2], sheer_coeffs_list[min_inx][3], (min_inx+1)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][4], sheer_coeffs_list[min_inx][5], (min_inx+2)%3)
                rot_obj = self._shear_transforms_for_3D_rot(rot_obj, sheer_coeffs_list[min_inx][6], sheer_coeffs_list[min_inx][7], (min_inx+0)%3)

        return rot_obj
        
    def _rotate_zxy_volume(
        self,
        volume_array,
        rot_matrix,
    ):
        """ """

        xp = self._xp
        affine_transform = self._affine_transform
        swap_zxy_to_xyz = self._swap_zxy_to_xyz

        volume = volume_array.copy()
        volume_shape = xp.asarray(volume.shape)
        tf = xp.asarray(swap_zxy_to_xyz.T @ rot_matrix.T @ swap_zxy_to_xyz)

        in_center = (volume_shape - 1) / 2
        out_center = tf @ in_center
        offset = in_center - out_center

        volume = affine_transform(volume, tf, offset=offset, order=3)

        return volume
    

    def rotate_3d(
        self,
        obj,
        rotmat
    ):
        """ """
        xp = self._xp
        
        if self._rot_method == 'Fourier_sheer':
            obj = xp.asarray(obj, dtype=xp.complex64)
            rot_obj = self._Fourier_sheer_rotate_3d(obj, rotmat)
        else:
            obj = xp.asarray(obj)
            rot_obj = self._rotate_zxy_volume(obj, rotmat)

        if self._object_type == 'potential':
            rot_obj = xp.real(rot_obj)
        return rot_obj
            
