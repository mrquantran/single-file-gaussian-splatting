import os
import json
import math
import random

import torch
import torch.nn as nn
import torchvision
import torchviz  # pip install torchviz

import numpy as np
from PIL import Image
from simple_knn._C import distCUDA2

from train_utils import update_learning_rate, ssim
from gauss_util import ply, obj
from gauss_rasterize.gauss_rasterize import GaussRasterizerSetting, GaussRasterizer

class GsDataset:
    def __init__(self, device, image_camera_path, resolution=(256, 256)):
        def getWorld2View2(
            R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0
        ) -> np.float32:
            """
            Computes the world-to-camera transformation matrix
            """
            Rt = np.zeros((4, 4))  # 4x4 matrix

            Rt[:3, :3] = R.transpose()
            Rt[:3, 3] = t
            Rt[3, 3] = 1.0

            # The camera-to-world matrix is computed as the inverse of the `Rt` matrix:
            C2W = np.linalg.inv(Rt)

            # Adjust the Camera Center
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center

            # Recompute World-to-Camera Matrix
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        def load_image_camera_from_transforms(
            device,
            path,
            resolution,
            transforms_file="transforms.json",
            white_background=False,
        ):
            """
            - The `load_image_camera_from_transforms` function reads images and corresponding camera data
            from a `transforms.json` file, processes the data, and constructs a list of `Camera` objects.
            - Each `Camera` object encapsulates image data, camera intrinsic and extrinsic parameters, and transformation matrices.
            """
            class Camera:
                """
                The `Camera` class is central to this function, as it encapsulates:
                    1. **Intrinsic Parameters:** Field of View (FoV), near and far plane distances.
                    2. **Extrinsic Parameters:** Rotation (`R`), translation (`t`), and transformation matrices.
                    3. **Projection Matrix:** Converts 3D points into the 2D image plane.
                """
                def __init__(
                    self,
                    device,
                    uid,
                    image_data,
                    image_path,
                    image_name,
                    image_width,
                    image_height,
                    R,
                    t,
                    FovX,
                    FovY,
                    znear=0.01,
                    zfar=100.0,
                    trans=np.array([0.0, 0.0, 0.0]),
                    scale=1.0,
                ):
                    def getProjectionMatrix(znear, zfar, fovX, fovY):
                        """
                        The projection matrix `P` maps 3D coordinates in camera space to 2D image coordinates, incorporating perspective distortion.
                        """
                        tanHalfFovY = math.tan((fovY / 2))
                        tanHalfFovX = math.tan((fovX / 2))

                        top = tanHalfFovY * znear
                        bottom = -top
                        right = tanHalfFovX * znear
                        left = -right

                        P = torch.zeros(4, 4)
                        z_sign = 1.0
                        P[0, 0] = 2.0 * znear / (right - left)
                        P[1, 1] = 2.0 * znear / (top - bottom)
                        P[0, 2] = (right + left) / (right - left)
                        P[1, 2] = (top + bottom) / (top - bottom)
                        P[3, 2] = z_sign
                        P[2, 2] = z_sign * zfar / (zfar - znear)
                        P[2, 3] = -(zfar * znear) / (zfar - znear)

                        return P

                    self.uid = uid
                    image_data = torch.from_numpy(np.array(image_data)) / 255.0
                    self.image_goal = (
                        image_data.clone().clamp(0.0, 1.0).permute(2, 0, 1).to(device)
                    )
                    self.image_tidy = (
                        image_data.permute(2, 0, 1)
                        if len(image_data.shape) == 3
                        else image_data.unsqueeze(dim=-1).permute(2, 0, 1)
                    )
                    self.image_path = image_path
                    self.image_name = image_name
                    self.image_width = image_width
                    self.image_height = image_height
                    self.R = R
                    self.t = t
                    self.FovX = FovX
                    self.FovY = FovY
                    self.znear = znear
                    self.zfar = zfar
                    self.trans = trans
                    self.scale = scale
                    self.world_view_transform = (
                        torch.tensor(getWorld2View2(R, t, self.trans, self.scale))
                        .transpose(0, 1)
                        .to(device)
                    )
                    self.full_proj_transform = (
                        self.world_view_transform.unsqueeze(0).bmm(
                            getProjectionMatrix(
                                znear=self.znear,
                                zfar=self.zfar,
                                fovX=self.FovX,
                                fovY=self.FovY,
                            )
                            .transpose(0, 1)
                            .to(device)
                            .unsqueeze(0)
                        )
                    ).squeeze(0)
                    self.camera_center = self.world_view_transform.inverse()[3, :3]

            def fov2focal(fov, pixels):
                return pixels / (2 * math.tan(fov / 2))

            def focal2fov(focal, pixels):
                return 2 * math.atan(pixels / (2 * focal))

            image_camera = []
            with open(os.path.join(path, transforms_file)) as json_file:
                transforms_json = json.load(json_file)
                fovx = transforms_json["camera_angle_x"]
                for idx, frame in enumerate(transforms_json["frames"]):
                    image_path = os.path.join(path, frame["file_path"])
                    image_norm = (
                        np.array(Image.open(image_path).convert("RGBA")) / 255.0
                    )
                    image_back = np.array(
                        (
                            np.array([1.0, 1.0, 1.0])
                            if white_background
                            else np.array([0.0, 0.0, 0.0])
                        )
                        * (1.0 - image_norm[:, :, 3:4])
                        * 255,
                        dtype=np.byte,
                    )
                    image_fore = np.array(
                        image_norm[:, :, :3] * image_norm[:, :, 3:4] * 255,
                        dtype=np.byte,
                    )
                    image_data = Image.fromarray(image_fore + image_back, "RGB").resize(resolution)

                    # NeRF 'transform_matrix' is a camera-to-world transform
                    c2w = np.array(frame["transform_matrix"])

                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    c2w[:3, 1:3] *= -1

                    # get the world-to-camera transform and set R, T
                    w2c = np.linalg.inv(c2w)

                    # R is stored transposed due to 'glm' in CUDA code
                    R, t = (np.transpose(w2c[:3, :3]), w2c[:3, 3])

                    fovy = focal2fov(fov2focal(fovx, image_data.size[0]), image_data.size[1])
                    camera = Camera(
                        device=device,
                        uid=idx,
                        image_data=image_data,
                        image_path=image_path,
                        image_name=os.path.basename(image_path),
                        image_width=image_data.size[0],
                        image_height=image_data.size[1],
                        R=R,
                        t=t,
                        FovX=fovx,
                        FovY=fovy,
                    )
                    image_camera.append(camera)

            return image_camera

        def getNerfppNorm(cam_info):
            """
            This code defines a function `getNerfppNorm` that normalizes camera positions for consistent scaling in a 3D scene,
            based on the **NeRF++ (Neural Radiance Fields++)** approach. It computes the scene's bounding sphere
            by calculating the centroid of all camera positions and the radius that bounds them.
            """
            def get_center_and_diag(cam_centers):
                """
                The diagonal of the bounding box is calculated as the maximum Euclidean distance from the bounding box center to any camera center.
                """
                cam_centers = np.hstack(cam_centers)
                avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
                center = avg_cam_center
                dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
                diagonal = np.max(dist)
                return center.flatten(), diagonal

            # The function iterates over the camera information (cam_info) to compute the centers of all cameras in world space.
            # Camera centers are extracted from the Camera-to-World transformation matrix (C2W), specifically the translation component.
            cam_centers = []
            for cam in cam_info:
                W2C = getWorld2View2(cam.R, cam.t)
                C2W = np.linalg.inv(W2C)
                cam_centers.append(C2W[:3, 3:4])
            center, diagonal = get_center_and_diag(cam_centers)

            # The diagonal distance is scaled by a factor (e.g., 1.1) to slightly extend the extent for robustness.
            radius = diagonal * 1.1
            translate = -center

            return {"translate": translate, "radius": radius}

        self.image_camera = load_image_camera_from_transforms(
            device, image_camera_path, resolution
        )
        self.cameras_extent = getNerfppNorm(self.image_camera)["radius"]


class GsNetwork(torch.nn.Module):
    def __init__(self, device, point_number, percent_dense=0.01, max_sh_degree=3):
        super().__init__()
        self.percent_dense = percent_dense
        self.max_sh_degree, self.now_sh_degree = max_sh_degree, 0  #spherical-harmonics

        points = (torch.rand(point_number, 3).float().to(device) - 0.5) * 1.0
        features = torch.cat((torch.rand(point_number, 3, 1).float().to(device) / 5.0 + 0.4, torch.zeros((point_number, 3, (self.max_sh_degree + 1) ** 2 -1)).float().to(device)), dim=-1)
        scale = torch.log(torch.sqrt(torch.clamp_min(distCUDA2(points).float(), 0.0000001)))[...,None].repeat(1, 3)  #torch.ones(point_number, 3).float().to(device)  #John
        rotation = torch.cat((torch.ones((point_number, 1)).float().to(device), torch.zeros((point_number, 3)).float().to(device)), dim=1)
        opacity = torch.log((torch.ones((point_number, 1)).float().to(device) * 0.1) / (1. - (torch.ones((point_number, 1)).float().to(device) * 0.1)))
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            def build_scaling_rotation(s, r):
                def build_rotation(r):
                    norm = torch.sqrt(
                        r[:, 0] * r[:, 0]
                        + r[:, 1] * r[:, 1]
                        + r[:, 2] * r[:, 2]
                        + r[:, 3] * r[:, 3]
                    )
                    q = r / norm[:, None]
                    R = torch.zeros((q.size(0), 3, 3), device="cuda")
                    r = q[:, 0]
                    x = q[:, 1]
                    y = q[:, 2]
                    z = q[:, 3]
                    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
                    R[:, 0, 1] = 2 * (x * y - r * z)
                    R[:, 0, 2] = 2 * (x * z + r * y)
                    R[:, 1, 0] = 2 * (x * y + r * z)
                    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
                    R[:, 1, 2] = 2 * (y * z - r * x)
                    R[:, 2, 0] = 2 * (x * z - r * y)
                    R[:, 2, 1] = 2 * (y * z + r * x)
                    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
                    return R

                L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
                R = build_rotation(r)
                L[:, 0, 0] = s[:, 0]
                L[:, 1, 1] = s[:, 1]
                L[:, 2, 2] = s[:, 2]
                L = R @ L
                return L

            def strip_symmetric(sym):
                def strip_lowerdiag(L):
                    uncertainty = torch.zeros(
                        (L.shape[0], 6), dtype=torch.float, device="cuda"
                    )
                    uncertainty[:, 0] = L[:, 0, 0]
                    uncertainty[:, 1] = L[:, 0, 1]
                    uncertainty[:, 2] = L[:, 0, 2]
                    uncertainty[:, 3] = L[:, 1, 1]
                    uncertainty[:, 4] = L[:, 1, 2]
                    uncertainty[:, 5] = L[:, 2, 2]
                    return uncertainty

                return strip_lowerdiag(sym)

            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize  # default 2
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = lambda x: torch.log(
            x / (1.0 - x)
        )  # inverse-sigmoid

        self.max_radii2D = torch.zeros((point_number)).float().to(device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

    def oneupSHdegree(self):
        if self.now_sh_degree < self.max_sh_degree:
            self.now_sh_degree += 1

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def densify_and_prune(
        self, max_grad, min_opacity, extent, max_screen_size, optimizer
    ):
        # Adds new Gaussians to the current set and updates the optimizer.
        def densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            optimizer,
        ):
            """
            For each parameter (e.g., positions xyz, opacity, scaling), concatenate the new Gaussian values (extension_tensor) to the existing tensors.
            Update the optimizer's internal state (exp_avg and exp_avg_sq) with zero-initialized values for the new Gaussians.
            """
            def cat_tensors_to_optimizer(tensors_dict, optimizer):
                optimizable_tensors = {}
                for group in optimizer.param_groups:
                    assert len(group["params"]) == 1
                    extension_tensor = tensors_dict[group["name"]]
                    stored_state = optimizer.state.get(group["params"][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat(
                            (
                                stored_state["exp_avg"],
                                torch.zeros_like(extension_tensor),
                            ),
                            dim=0,
                        )
                        stored_state["exp_avg_sq"] = torch.cat(
                            (
                                stored_state["exp_avg_sq"],
                                torch.zeros_like(extension_tensor),
                            ),
                            dim=0,
                        )
                        del optimizer.state[group["params"][0]]
                        group["params"][0] = nn.Parameter(
                            torch.cat(
                                (group["params"][0], extension_tensor), dim=0
                            ).requires_grad_(True)
                        )
                        optimizer.state[group["params"][0]] = stored_state
                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(
                            torch.cat(
                                (group["params"][0], extension_tensor), dim=0
                            ).requires_grad_(True)
                        )
                        optimizable_tensors[group["name"]] = group["params"][0]
                return optimizable_tensors

            d = {
                "xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacities,
                "scaling": new_scaling,
                "rotation": new_rotation,
            }
            # Replace existing tensors (e.g., _xyz, _opacity) with the updated tensors that now include the new Gaussians.
            optimizable_tensors = cat_tensors_to_optimizer(d, optimizer)

            # Reinitialize gradient accumulators (xyz_gradient_accum) and other related metrics (denom, max_radii2D) for the new set of Gaussians.
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self.xyz_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Removes Gaussians that meet pruning criteria.
        def prune_points(mask, optimizer):
            def _prune_optimizer(mask, optimizer):
                optimizable_tensors = {}
                for group in optimizer.param_groups:
                    stored_state = optimizer.state.get(group["params"][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                        del optimizer.state[group["params"][0]]
                        group["params"][0] = nn.Parameter(
                            (group["params"][0][mask].requires_grad_(True))
                        )
                        optimizer.state[group["params"][0]] = stored_state
                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(
                            group["params"][0][mask].requires_grad_(True)
                        )
                        optimizable_tensors[group["name"]] = group["params"][0]
                return optimizable_tensors

            # Use a Boolean mask (valid_points_mask) to select the points to keep (~mask).
            valid_points_mask = ~mask

            # Apply the mask to each parameter tensor (e.g., _xyz, _opacity, _scaling) and update the optimizer's tensors (exp_avg, exp_avg_sq) to reflect the reduced size.
            optimizable_tensors = _prune_optimizer(valid_points_mask, optimizer)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Adds new Gaussians by cloning existing ones that meet a gradient-based threshold.
        def densify_and_clone(grads, grad_threshold, scene_extent, optimizer):
            # Select Points to Clone:
            # Identify points with gradients above grad_threshold (selected_pts_mask).
            selected_pts_mask = torch.where(
                torch.norm(grads, dim=-1) >= grad_threshold, True, False
            )

            # Ensure these points are within a reasonable scene extent (scene_extent).
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values
                <= self.percent_dense * scene_extent,
            )

            # Extract positions (_xyz), features, opacity, scaling, and rotation for the selected points.
            # Pass these as new Gaussians to densification_postfix.
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacities,
                new_scaling,
                new_rotation,
                optimizer,
            )

        # Adds new Gaussians by splitting existing ones that meet a different gradient-based threshold.
        def densify_and_split(grads, grad_threshold, scene_extent, optimizer, N=2):
            def build_rotation(r):
                norm = torch.sqrt(
                    r[:, 0] * r[:, 0]
                    + r[:, 1] * r[:, 1]
                    + r[:, 2] * r[:, 2]
                    + r[:, 3] * r[:, 3]
                )
                q = r / norm[:, None]
                R = torch.zeros((q.size(0), 3, 3), device=r.device)
                r = q[:, 0]
                x = q[:, 1]
                y = q[:, 2]
                z = q[:, 3]
                R[:, 0, 0] = 1 - 2 * (y * y + z * z)
                R[:, 0, 1] = 2 * (x * y - r * z)
                R[:, 0, 2] = 2 * (x * z + r * y)
                R[:, 1, 0] = 2 * (x * y + r * z)
                R[:, 1, 1] = 1 - 2 * (x * x + z * z)
                R[:, 1, 2] = 2 * (y * z - r * x)
                R[:, 2, 0] = 2 * (x * z - r * y)
                R[:, 2, 1] = 2 * (y * z + r * x)
                R[:, 2, 2] = 1 - 2 * (x * x + y * y)
                return R

            # Select Points to Split:
            # Identify points with gradients above grad_threshold and
            # scaling values that exceed a certain percentage of the scene extent (percent_dense).
            n_init_points = self.get_xyz.shape[0]
            padded_grad = torch.zeros((n_init_points), device=self.get_xyz.device)
            padded_grad[: grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values
                > self.percent_dense * scene_extent,
            )

            # Generate Split Points:
            # Use the Gaussian scaling as the standard deviation (stds) to sample new positions around the original points.
            # Apply random rotations to the sampled points (build_rotation).
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device=stds.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

            # Update Gaussian Properties: Compute scaling, rotation, opacity, and features for the split points.
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
                selected_pts_mask
            ].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

            # Add Split Gaussians: Pass the newly created Gaussians to densification_postfix.
            densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                optimizer,
            )

            # Prune Originals: Mark the original points for pruning after splitting.
            prune_filter = torch.cat(
                (
                    selected_pts_mask,
                    torch.zeros(
                        N * selected_pts_mask.sum(),
                        device=selected_pts_mask.device,
                        dtype=bool,
                    ),
                )
            )
            prune_points(prune_filter, optimizer)

        # Compute the average gradient for each Gaussian using accumulated statistics.
        # Handle any invalid values by setting them to 0.
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone Gaussians with gradients above max_grad and scaling below a threshold (percent_dense * scene_extent).
        densify_and_clone(grads, max_grad, extent, optimizer)

        # Split Gaussians with gradients above max_grad and scaling above a threshold (percent_dense * scene_extent).
        densify_and_split(grads, max_grad, extent, optimizer)

        # Pruning Conditions:
        # - Low Opacity: Remove Gaussians with opacity below min_opacity.
        # - Large Screen-Space Radius: Remove Gaussians with radii exceeding max_screen_size.
        # - Large Scaling in World Space: Remove Gaussians with scaling exceeding 10% of the scene extent.
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        prune_points(prune_mask, optimizer)
        torch.cuda.empty_cache()

    def reset_opacity(self, optimizer):
        def replace_tensor_to_optimizer(tensor, name, optimizer):
            optimizable_tensors = {}
            for group in optimizer.param_groups:
                if group["name"] == name:
                    stored_state = optimizer.state.get(group["params"][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
            return optimizable_tensors

        def inverse_sigmoid(x):
            return torch.log(x / (1.0 - x))

        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        self._opacity = replace_tensor_to_optimizer(
            opacities_new, "opacity", optimizer
        )["opacity"]

    def get_covariance(self, scaling_modifier=1):  # call in render, must have
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )  # build_covariance_from_scaling_rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # exp

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)  # normalize  #default 2

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)  # sigmoid

class GsRender:
    def render(
        self, viewpoint_camera, pc, bg_color, device, scale_modifier=1.0, is_train=True
    ):
        screenspace_points = torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=is_train, device=device
        )
        rasterizer = GaussRasterizer(
            setting=GaussRasterizerSetting(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=math.tan(viewpoint_camera.FovX * 0.5),
                tanfovy=math.tan(viewpoint_camera.FovY * 0.5),
                scale_modifier=scale_modifier,
                sh_degree=pc.now_sh_degree,
                prefiltered=False,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                campos=viewpoint_camera.camera_center,
                bg=bg_color,
            )
        )
        rendered_image, radii = rasterizer(
            means3D=pc.get_xyz,
            means2D=screenspace_points,
            opacities=pc.get_opacity,
            shs=pc.get_features,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
        )
        return rendered_image, screenspace_points, radii


def make(device, spatial_lr_scale=1.0, position_lr_max_steps= 1000 * 10):
    # Loads images and corresponding camera parameters from the given path.
    gsDataset = GsDataset(device=device, image_camera_path="./data/image/wizard/")

    # Initialize the parametrs of Gaussian 3D
    # Point_number will get from Colmap
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/__init__.py#L44
    gsNetwork = GsNetwork(device=device, point_number= 1 * 10000)

    # A custom renderer to project 3D Gaussian splats into 2D images using CUDA acceleration.
    gsRender = GsRender()

    optimizer = torch.optim.Adam(
        [
            {
                "params": [gsNetwork._xyz],
                "lr": 0.00016 * spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [gsNetwork._features_dc], "lr": 0.0025, "name": "f_dc"},
            {
                "params": [gsNetwork._features_rest],
                "lr": 0.0025 / 20.0,
                "name": "f_rest",
            },
            {"params": [gsNetwork._opacity], "lr": 0.05, "name": "opacity"},
            {"params": [gsNetwork._scaling], "lr": 0.005, "name": "scaling"},
            {"params": [gsNetwork._rotation], "lr": 0.001, "name": "rotation"},
        ],
        lr=0.0,
        eps=1e-15,
    )

    # Densification: Adjusts the number of 3D Gaussian based on the gradient of the rendered image
    # Opacity Reset: Periodically resets opacity
    densification_interval = position_lr_max_steps // 300
    opacity_reset_interval = position_lr_max_steps // 10
    densify_from_iter = position_lr_max_steps // 6
    densify_until_iter = position_lr_max_steps // 2
    densify_grad_threshold = 0.0002
    densify_opacity_threshold = 0.005

    loss_weight_L1 = 0.8
    loss_weight_dssim = 0.2

    white_background = 0
    background = (torch.tensor([[0, 0, 0], [1, 1, 1]][white_background]).float().to(device))

    viewpoint_stack: list = gsDataset.image_camera.copy() # get 2D image list from dataset

    for iteration in range(1, position_lr_max_steps + 1):
        if iteration % (position_lr_max_steps // 30) == 0:
            gsNetwork.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[random.randint(0, len(viewpoint_stack) - 1)]
        gt_image = viewpoint_cam.image_goal

        image, viewspace_point_tensor, radii = gsRender.render(
            viewpoint_cam, gsNetwork, background, device=device
        )

        # Radii: represents the computed radius of each Gaussian point in screen space.
        # Identifies which Gaussians have a positive radius and are thus visible
        visibility_filter = radii > 0

        L1 = torch.abs((image - gt_image)).mean()
        DSSIM = 1.0 - ssim(image, gt_image)

        loss = (
            loss_weight_L1 * L1
            + loss_weight_dssim * DSSIM
        )

        if iteration == 1:
            torchviz.make_dot(image).render(
                filename="network_image",
                directory="./nets/",
                format="svg",
                view=False,
                cleanup=True,
                quiet=True,
            )
            torchviz.make_dot(loss).render(
                filename="network_loss",
                directory="./nets/",
                format="svg",
                view=False,
                cleanup=True,
                quiet=True,
            )
        loss.backward()

        with torch.no_grad():
            # use no_grad to disable automatic gradient-descent, and control it by self.
            # xyz,clone/split/prune,opacity,...
            # else just optimize parameters in render: screenspace_points @ render, opacity, scale, rotation, feature
            if (iteration < densify_until_iter):
                # Identify significant Gaussians (large radii) and those that can potentially be pruned (small radii).
                gsNetwork.max_radii2D[visibility_filter] = torch.max(
                    gsNetwork.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                # Accumulates gradient-related statistics for visible Gaussians.
                # used to decide where to add new Gaussians during densification.
                gsNetwork.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                # After a certain number of iterations (densify_from_iter).
                # At regular intervals defined by densification_interval.
                if (iteration > densify_from_iter and iteration % densification_interval == 0):
                    # Limits the maximum allowable size of Gaussians for pruning decisions.
                    max_screen_size_threshold = (
                        16 if iteration > opacity_reset_interval else None
                    )
                    gsNetwork.densify_and_prune(
                        densify_grad_threshold,
                        densify_opacity_threshold,
                        gsDataset.cameras_extent,
                        max_screen_size_threshold,
                        optimizer,
                    )

                # Reinitializes opacity values to a better range, allowing gradients to flow smoothly again.
                if iteration > 0 and (
                    iteration % opacity_reset_interval == 0
                    or (white_background and iteration == densify_from_iter)
                ):
                    # opacity activation is sigmoid, so gradient_descent is inverse_sigmoid
                    gsNetwork.reset_opacity(
                        optimizer
                    )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        update_learning_rate(
            optimizer=optimizer,
            iteration=iteration,
            position_lr_max_steps=position_lr_max_steps,
            spatial_lr_scale=spatial_lr_scale,
        )

        if iteration % 100 == 0:
            print(
                "iteration=%06d/%06d  loss=%.6f"
                % (iteration, position_lr_max_steps, loss.item())
            )
            os.makedirs("./outs/shot/img/", exist_ok=True)

            torchvision.utils.save_image(
                image, "./outs/shot/img/image_%06d_o.png" % (iteration)
            )
            torchvision.utils.save_image(
                gt_image, "./outs/shot/img/image_%06d_t.png" % (iteration)
            )

            ply.save_ply(
                gsNetwork._xyz.detach().cpu(),
                gsNetwork._features_dc.detach().cpu(),
                gsNetwork._features_rest.detach().cpu(),
                gsNetwork._opacity.detach().cpu(),
                gsNetwork._scaling.detach().cpu(),
                gsNetwork._rotation.detach().cpu(),
                "./outs/shot/ply/ply_%06d_o.ply" % (iteration),
            )

    return gsNetwork

def mesh(gsNetwork, opacity_threshold, density_threshold):
    obj.save_mesh(
        gsNetwork,
        GsRender(),
        opacity_threshold=opacity_threshold,
        density_threshold=density_threshold,
        resolution=128,
        decimate_target=1 * 10000,
        texture_size=1024,
    )

def main(
    checkpoint="./outs/ckpt/checkpoint.pth",
    device=["cpu", "cuda"][torch.cuda.is_available()],
):
    if not os.path.exists(checkpoint):
        gsNetwork = make(device, position_lr_max_steps=1000 * 2)
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save({"network": gsNetwork.state_dict()}, checkpoint)
    else:
        gsCheckpoint = torch.load(checkpoint)["network"]
        gsNetwork = GsNetwork(
            device=device, point_number=gsCheckpoint["_xyz"].shape[0]
        ).to(device)
        gsNetwork.load_state_dict(gsCheckpoint)

    mesh(gsNetwork, opacity_threshold=0.001, density_threshold=0.333)

if __name__ == "__main__":  # python -Bu stereo_gauss_splat.py
    main()
