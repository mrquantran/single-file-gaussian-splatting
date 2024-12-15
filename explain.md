# Gaussian workflow
Certainly! Let's delve into the code step by step, explaining the mathematical concepts and how they relate to the 3D Gaussian Splatting paper. We'll focus on the Gaussian workflow, especially the cloning and splitting of 3D Gaussians, as implemented in the provided `stereo_gauss_splat.py` code.

---
### **1. Overview**
The goal of 3D Gaussian Splatting is to represent a 3D scene using a set of oriented 3D Gaussian functions, each characterized by its position, covariance (shape and orientation), color, and opacity. By optimizing these parameters, we can render images from new viewpoints that match the input images. The code implements this by:
- Loading image data and camera parameters.
- Initializing a set of 3D Gaussians (point cloud).
- Rendering images from different viewpoints.
- Computing losses and optimizing the parameters.
- Dynamically adjusting the Gaussians through cloning and splitting.
---
### **2. Data Loading (`GsDataset` Class)**
**Purpose**: Load images and corresponding camera parameters to prepare the dataset for training.

**Key Functions**:
- **`load_image_camera_from_transforms`**: Parses a `transforms.json` file containing camera parameters and loads images.
- **`getWorld2View2`**: Computes the world-to-camera transformation matrix.
- **`getProjectionMatrix`**: Computes the camera's projection matrix.

**Mathematical Details**:
1. **World-to-Camera Transformation**:
   Given camera rotation **R** and translation **t**, the world-to-camera transformation matrix is computed as: $$\mathbf{Rt} = \begin{bmatrix} \mathbf{R}^T & \mathbf{t} \\\mathbf{0}^T & 1 \end{bmatrix}$$Here, $\mathbf{R}^T$ is the transpose of the rotation matrix, and $\mathbf{t}$ is the translation vector.
2. **Camera-to-World Transformation**:
$$
 \mathbf{C2W} = \mathbf{Rt}^{-1}
$$
3. **Projection Matrix**: With near and far clipping planes $z_{near}$ and $z_{far}$, and field of view angles $(\text{fovX}, \text{fovY})$, the projection matrix **P** is computed.
4. **Adjusting Axes**: The code adjusts the axes from OpenGL/Blender conventions to the conventions used in COLMAP.
---
### **3. Network Initialization (`GsNetwork` Class)**
**Purpose**: Initialize the parameters of the 3D Gaussians, including positions, features, scaling, rotation, and opacity.

**Key Parameters**:
- **Positions $\mathbf{x}_i$**: Randomly initialized in the space $[-0.5, 0.5]^3$.
- **Features $\mathbf{f}_i$**: Includes color and spherical harmonics coefficients.
- **Scaling $\mathbf{s}_i$**: Defines the standard deviations along each axis, representing the size of the Gaussian.
- **Rotation $\mathbf{q}_i$**: Quaternions representing the orientation of the Gaussian.
- **Opacity $\alpha_i$**: Controls the transparency of the Gaussian.

**Mathematical Details**:
1. **Positions**: $$\mathbf{x}_i \sim \text{Uniform}(-0.5, 0.5)$$
2. **Features (Color and SH Coefficients)**:  The features are initialized with RGB values and higher-order spherical harmonics set to zero.
3. **Scaling**:  The initial scaling is based on the logarithm of the inter-point distances: $$\mathbf{s}_i = \log\left( \sqrt{\max\left( \|\mathbf{x}_i - \mathbf{x}_j\|^2, \epsilon \right)} \right)$$where $\epsilon$ is a small constant to prevent taking the log of zero.
4. **Rotation**: Initialized as identity quaternions representing no rotation: $$\mathbf{q}_i = [1, 0, 0, 0]$$
5. **Opacity**:  Initialized using the inverse sigmoid function: $$\alpha_i = \sigma^{-1}(0.1) = \log\left( \frac{0.1}{1 - 0.1} \right)$$where $\sigma^{-1}$ is the inverse sigmoid function.
---
### **4. Covariance Matrix Construction**
Each 3D Gaussian is characterized by a covariance matrix $\boldsymbol{\Sigma}_i$, encoding its size and orientation.

**Key Function**:  **`build_covariance_from_scaling_rotation`**: Constructs the covariance matrix for each Gaussian.

**Mathematical Details**:
1. **Scaling Matrix $\mathbf{D}_i$**:  A diagonal matrix with scaling factors:
$$
\mathbf{D}_i = \text{diag}(e^{\mathbf{s}_i})
$$
2. **Rotation Matrix $\mathbf{R}_i$**: Constructed from quaternions $\mathbf{q}_i$: $$\mathbf{R}_i = \text{QuaternionToRotationMatrix}(\mathbf{q}_i)$$
3. **Transformation Matrix $\mathbf{L}_i$**: Combines scaling and rotation:
$$
\mathbf{L}_i = \mathbf{R}_i \mathbf{D}_i
$$
4. **Covariance Matrix**: $$\boldsymbol{\Sigma}_i = \mathbf{L}_i \mathbf{L}_i^T$$
5. **Parameterization**:  To reduce redundancy, only the unique elements of the symmetric covariance matrix are stored: $$\text{covariance\_vector}_i = \left[ \Sigma_{xx}, \Sigma_{xy}, \Sigma_{xz}, \Sigma_{yy}, \Sigma_{yz}, \Sigma_{zz} \right]$$
---
### **5. Rendering (`GsRender` Class)**
**Purpose**: Render the scene by projecting the 3D Gaussians onto the 2D image plane, considering their properties.

**Key Steps**:
1. **Project Gaussians to Screen Space**:  Transform the 3D positions using the camera's view and projection matrices.
2. **Rasterization**:  Use the **GaussRasterizer** to render each Gaussian onto the image, considering:
   - The projected mean position.
   - The covariance matrix (defining the ellipse in screen space).
   - The features (color and SH coefficients).
   - The opacity.
3. **Composite the Gaussians**:  Blend the contributions of all Gaussians to form the final image.

**Mathematical Details**:
1. **Projection**:  The position of each Gaussian is transformed to screen space:  $$
  \mathbf{p}_i = \mathbf{P} \mathbf{V} \mathbf{x}_i$$where $\mathbf{V}$ is the view matrix, and $\mathbf{P}$ is the projection matrix.
2. **Screen-Space Covariance**:  The covariance matrices are transformed accordingly.
3. **Rasterization**:  For each pixel, the contribution from nearby Gaussians is computed, typically using a fragment shader-like approach.
---
### **6. Loss Computation and Optimization**
**Purpose**: Compute the loss between the rendered image and the ground truth, and optimize the Gaussian parameters.

**Key Steps**:
1. **Loss Function**: The loss is a combination of:
   - **L1 Loss**: $$\mathcal{L}_{\text{L1}} = \frac{1}{N_{\text{pixels}}} \sum_{\text{pixels}} |I_{\text{rendered}} - I_{\text{ground truth}}|$$
   - **Structural Similarity Index (SSIM) Loss**:  $$\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(I_{\text{rendered}}, I_{\text{ground truth}})  $$
   - **The total loss:**  $$\mathcal{L} = w_{\text{L1}} \mathcal{L}_{\text{L1}} + w_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}$$with weights $w_{\text{L1}}$ and $w_{\text{SSIM}}$.
2. **Optimization**: Use gradient descent to update the parameters:  $$ \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}  $$where $\theta$ represents all the parameters, and $\eta$ is the learning rate.
---
### **7. Dynamic Adjustment of Gaussians (Cloning and Splitting)**
**Purpose**: Adaptively refine the set of Gaussians to better represent detailed regions by cloning and splitting Gaussians based on specific criteria.

**Key Functions**:
- **`densify_and_prune`**: Main function to handle cloning and splitting.
- **`add_densification_stats`**: Collects gradient statistics for each Gaussian.
- **`densify_and_clone`**: Cloning mechanism.
- **`densify_and_split`**: Splitting mechanism.

**Mathematical Details**:
1. **Gradient Accumulation**:  Accumulate the gradients of the positions in screen space:  $$
   \text{grad\_norm}_i += \| \nabla_{\mathbf{x}_i} \mathcal{L} \|
  $$
2. **Cloning Criteria**: If a Gaussian has:
   - High gradient norm $text{grad\_norm}_i \geq \tau_{\text{clone}}$
   - Small scaling $\max(\mathbf{s}_i) \leq \gamma_{\text{scale}}$

   **Cloning Procedure**: Duplicate the Gaussian $$\left\{ \mathbf{x}_i, \mathbf{f}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i \right\} \rightarrow \text{copy}$$
3. **Splitting Criteria**: If a Gaussian has:
   - High gradient norm $\text{grad\_norm}_i \geq \tau_{\text{split}}$
   - Large scaling $\max(\mathbf{s}_i) > \gamma_{\text{scale}}$

   **Splitting Procedure**:
   1. Generate $N$ new Gaussians around $\mathbf{x}_i$:
     - Sample new positions:  $$\mathbf{x}'_i = \mathbf{x}_i + \Delta \mathbf{x}, \quad \Delta \mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{S}_i)$$
     - Adjust scaling: $$\mathbf{s}'_i = \log\left( e^{\mathbf{s}_i} / (0.8N) \right)$$
     - Copy other parameters: $$\left\{ \mathbf{f}'_i, \mathbf{q}'_i, \alpha'_i \right\} = \left\{ \mathbf{f}_i, \mathbf{q}_i, \alpha_i \right\}$$
   2. Remove the original Gaussian.

4. **Pruning Criteria**:
   - Opacity below a threshold $\alpha_i < \tau_{\text{opacity}}$
   - Excessive screen space size.

   **Pruning Procedure**:  Remove the Gaussian from the set.

**Rationale**:
- **Cloning** focuses on areas where small Gaussians need to cover regions with high detail, but the Gaussians are already small.
- **Splitting** is for larger Gaussians that need to be subdivided to capture finer details.
---
### **8. Training Loop (`make` Function)**
**Purpose**: Run the optimization over multiple iterations, periodically adjusting the Gaussians.

**Key Steps**:
1. **Iterations**: Loop over a fixed number of steps.
2. **Viewpoint Selection**:  Randomly select a camera viewpoint for each iteration.
3. **Rendering and Loss Computation**:  Render the image and compute the loss.
4. **Backpropagation**: Compute gradients.
5. **Parameter Update**: Update parameters using the optimizer.
6. **Densification and Pruning**: Periodically call `densify_and_prune` based on iteration counts.
7. **Learning Rate Scheduling**: Adjust the learning rate over time.
---
### **9. Saving and Loading Model (`main` Function)**
**Purpose**: Save the trained model to a checkpoint or load from a checkpoint for further processing.

---
### **10. Mesh Extraction**
**Purpose**: Extract a mesh from the trained Gaussians for visualization or further processing.

**Key Function**: **`mesh`**: Invokes methods to save the Gaussians as a mesh.

**Process**: Convert the set of Gaussians into a mesh representation, possibly using techniques like marching cubes.
# Adaptive Density Control
### **1. Introduction to Adaptive Density Control**
In 3D Gaussian Splatting, we represent a scene with a set of oriented 3D Gaussians, each with parameters encoding its position, shape, orientation, color, and opacity. However, the initial distribution of Gaussians might not be sufficient to capture all the scene details, especially in regions requiring higher resolution.

**Adaptive Density Control** aims to adjust the distribution of Gaussians dynamically during training by **cloning** and **splitting** existing Gaussians based on specific criteria. This allows the model to concentrate computational resources where they are most needed.

---
### **2. Overview of Relevant Classes and Functions**
  Before diving into the details, let's identify the key parts of the code related to dynamic adjustment:
- **Class `GsNetwork`**:
  - Maintains the parameters of the Gaussians.
  - Contains methods for cloning and splitting.
- **Functions within `GsNetwork`**:
  - **`add_densification_stats`**: Collects gradient statistics for each Gaussian.
  - **`densify_and_prune`**: Main function orchestrating cloning, splitting, and pruning.
  - **`densify_and_clone`**: Implements the cloning mechanism.
  - **`densify_and_split`**: Implements the splitting mechanism.
  - **`prune_points`**: Removes Gaussians that meet certain pruning criteria.
---
### **3. Collecting Gradient Statistics**
#### **Function: `add_densification_stats`**
**Purpose**: Accumulate gradient norms associated with each Gaussian to guide cloning and splitting decisions.

**Code Snippet**:
```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
    )
    self.denom[update_filter] += 1
```

**Explanation**:
- **`viewspace_point_tensor`**: The 3D positions of Gaussians in view space (after transformation by the camera's view matrix).
- **`update_filter`**: A boolean mask indicating which Gaussians are visible in the current view (i.e., contribute to the rendered image).
- **Gradient Accumulation**: For each visible Gaussian \( i \), we accumulate the norm of its position gradient in screen space: $$\text{grad\_norm}_i += \left\| \nabla_{\mathbf{x}_i^{\text{vs}}} \mathcal{L} \right\|$$where $mathbf{x}_i^{\text{vs}}$ is the position of Gaussian $i$ in view space, and $\mathcal{L}$ is the loss function.
- **Denominator Update**:  The **`denom`** variable counts how many times each Gaussian has been updated (i.e., appeared in the rendered images).
---
### **4. Dynamic Adjustment: Cloning and Splitting**
#### **Function: `densify_and_prune`**
**Purpose**: Based on accumulated gradient statistics and other criteria, decide which Gaussians to clone, split, or prune.

**Code Snippet**:
```python
def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, optimizer):
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0

    # Cloning
    densify_and_clone(grads, max_grad, extent, optimizer)

    # Splitting
    densify_and_split(grads, max_grad, extent, optimizer)

    # Pruning
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    prune_points(prune_mask, optimizer)
```

**Explanation**:
1. **Gradient Averaging**: Compute the average gradient norm for each Gaussian: $$
\overline{\text{grad}}_i = \frac{\text{grad\_norm}_i}{\text{denom}_i}
$$This represents how much each Gaussian contributes to the loss over the iterations.
2. **Cloning**:
   - **Criteria**: Clone Gaussians that meet both:
     - **High Gradient Norm**:  $$\overline{\text{grad}}_i \geq \tau_{\text{clone}}  $$
     - **Small Scale**: $$ \max(\mathbf{s}_i) \leq \gamma_{\text{scale}} \cdot \text{scene\_extent}  $$where $\gamma_{\text{scale}}$ is a small fraction (e.g., 0.01).
   - **Rationale**: Gaussians with small sizes and high gradient norms indicate regions where more detail is needed, but the Gaussians cannot get smaller.

3. **Splitting**:
   - **Criteria**: Split Gaussians that meet both:
     - **High Gradient Norm**: $$ \overline{\text{grad}}_i \geq \tau_{\text{split}}  $$
     - **Large Scale**: $$ \max(\mathbf{s}_i) > \gamma_{\text{scale}} \cdot \text{scene\_extent}  $$
   - **Rationale**: Large Gaussians with high gradient norms suggest that subdividing them into smaller Gaussians can help capture finer details.

4. **Pruning**:
   - **Criteria**: Remove Gaussians that:
     - **Low Opacity**:  $$\alpha_i < \tau_{\text{opacity}}$$
     - **Excessive Screen Space Size**: $$\text{radii2D}_i > \text{max\_screen\_size}$$
     - **Excessive World Space Size**: $$ \max(\mathbf{s}_i) > 0.1 \times \text{scene\_extent}  $$
---
#### **Function: `densify_and_clone`**
**Purpose**: Duplicate Gaussians meeting the cloning criteria to increase density in high-gradient, fine-detail areas.

**Code Snippet**:
```python
def densify_and_clone(grads, grad_threshold, scene_extent, optimizer):
    selected_pts_mask = torch.where(
        torch.norm(grads, dim=-1) >= grad_threshold, True, False
    )
    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
    )
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
```

**Mathematical Explanation**:
1. **Selection Criteria**: Gaussians to clone satisfy:
   - High average gradient norm: $\overline{\text{grad}}_i \geq \tau_{\text{clone}}$
   - Small scale relative to scene extent: $\max(\mathbf{s}_i) \leq \gamma_{\text{scale}} \cdot \text{scene\_extent}$
2. **Cloning Procedure**: For each selected Gaussian $i$ Duplicate its parameters: $$\begin{aligned}\mathbf{x}_{i, \text{new}} &= \mathbf{x}_i \\\mathbf{f}_{i, \text{new}} &= \mathbf{f}_i \\\mathbf{s}_{i, \text{new}} &= \mathbf{s}_i \\\mathbf{q}_{i, \text{new}} &= \mathbf{q}_i \\\alpha_{i, \text{new}} &= \alpha_i\end{aligned}$$This effectively doubles the density in that region.
---
#### **Function: `densify_and_split`**
**Purpose**: Replace Gaussians meeting the splitting criteria with multiple smaller Gaussians to better capture detail.

**Code Snippet**:
```python
def densify_and_split(grads, grad_threshold, scene_extent, optimizer, N=2):
    selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
    )
    stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3), device=stds.device)
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
    new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
    new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
    new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
    new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
    densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        optimizer,
    )
    prune_filter = torch.cat(
        (
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device=selected_pts_mask.device, dtype=bool),
        )
    )
    prune_points(prune_filter, optimizer)
```

**Mathematical Explanation**:
1. **Selection Criteria**: Gaussians to split satisfy:
   - High average gradient norm: $\overline{\text{grad}}_i \geq \tau_{\text{split}}$
   - Large scale relative to scene extent: $\max(\mathbf{s}_i) > \gamma_{\text{scale}} \cdot \text{scene\_extent}$
2. **Splitting Procedure**: For each selected Gaussian \( i \), generate \( N \) new Gaussians:
   - **Sample Offsets**: Draw samples from a normal distribution centered at zero with standard deviation equal to the scaling factors: $\Delta \mathbf{x}_{i, k} \sim \mathcal{N}(\mathbf{0}, \mathbf{S}_i)$ for \( k = 1, \dots, N \).
   - **Rotate Offsets**: Apply the rotation matrix $\mathbf{R}_i$ to the offsets: $\Delta \mathbf{x}'_{i, k} = \mathbf{R}_i \Delta \mathbf{x}_{i, k}$
   - **Compute New Positions**: New positions: $\mathbf{x}_{i, k, \text{new}} = \mathbf{x}_i + \Delta \mathbf{x}'_{i, k}$
   - **Adjust Scaling**: New scaling factors are reduced: $\mathbf{s}_{i, \text{new}} = \ln\left( \frac{e^{\mathbf{s}_i}}{0.8 N} \right)$ Specifically, the scales are reduced by a factor proportional to \( N \) to prevent the overall volume from increasing.
   - **Duplicate Other Parameters**: $$\begin{aligned}
     \mathbf{f}_{i, k, \text{new}} &= \mathbf{f}_i \\
     \mathbf{q}_{i, k, \text{new}} &= \mathbf{q}_i \\
     \alpha_{i, k, \text{new}} &= \alpha_i
     \end{aligned}$$
3. **Pruning Original Gaussians**: After creating the new Gaussians, the original Gaussians are removed to avoid redundancy.
---
#### **Function: `densification_postfix`**
**Purpose**: Integrate the newly created Gaussians into the model's parameters and optimizer's state.

**Code Snippet**:
```python
def densification_postfix(
    new_xyz,
    new_features_dc,
    new_features_rest,
    new_opacities,
    new_scaling,
    new_rotation,
    optimizer,
):
    d = {
        "xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling": new_scaling,
        "rotation": new_rotation,
    }
    optimizable_tensors = cat_tensors_to_optimizer(d, optimizer)
    self._xyz = optimizable_tensors["xyz"]
    self._features_dc = optimizable_tensors["f_dc"]
    self._features_rest = optimizable_tensors["f_rest"]
    self._opacity = optimizable_tensors["opacity"]
    self._scaling = optimizable_tensors["scaling"]
    self._rotation = optimizable_tensors["rotation"]
    self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```

**Explanation**:
- **Purpose**:
  - Update the model's parameters by concatenating the new Gaussians with the existing ones.
  - Update the optimizer's state to include the new parameters.
  - Reset gradient accumulators and denominators since the set of Gaussians has changed.
- **`cat_tensors_to_optimizer` Function**:
  - For each parameter group in the optimizer, extend the tensors with the new values.
  - Ensure that optimizer states (e.g., moment estimates in Adam) are appropriately extended with zeros for the new parameters.
---
#### **Function: `prune_points`**
**Purpose**: Remove Gaussians that meet pruning criteria from the model and the optimizer.

**Code Snippet**:
```python
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

    valid_points_mask = ~mask
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
```

**Explanation**:
- **Purpose**: Efficiently remove Gaussians from the model's parameters and the optimizer's state when they are deemed unnecessary.
- **Procedure**:
  - Create a mask (`valid_points_mask`) to select Gaussians to keep.
  - For each parameter group in the optimizer:
    - Update the optimizer's state (e.g., moment estimates) to reflect only the remaining Gaussians.
    - Update the parameters by selecting the entries corresponding to `valid_points_mask`.
---
### **5. Integration in the Training Loop**
The cloning and splitting mechanisms are integrated into the training loop to adaptively refine the Gaussian representation.

#### **Training Loop Snippet**:
```python
for iteration in range(1, position_lr_max_steps + 1):
    # ... Training steps ...

    # Accumulate gradient statistics
    gsNetwork.add_densification_stats(
        viewspace_point_tensor, visibility_filter
    )

    # Possibly perform densification
    if (
        iteration > densify_from_iter
        and iteration % densification_interval == 0
    ):
        gsNetwork.densify_and_prune(
            densify_grad_threshold,
            densify_opacity_threshold,
            gsDataset.cameras_extent,
            max_screen_size_threshold,
            optimizer,
        )
```

**Explanation**:
- **Gradient Accumulation**: During each iteration, after backpropagation, the gradient norms are accumulated for visible Gaussians.
- **Densification Checks**:
  - After a certain number of iterations (`densify_from_iter`), densification (`densify_and_prune`) is performed at regular intervals (`densification_interval`).
  - This allows enough time for the gradient statistics to accumulate before making cloning or splitting decisions.
---
### **7. Mathematical Summary**
- **Gradient Norms**: Average gradient norm per Gaussian: $\overline{\text{grad}}_i = \frac{\sum_k \left\| \nabla_{\mathbf{x}_i^{\text{vs}}} \mathcal{L}_k \right\|}{N_i}$ where \( N_i \) is the number of times Gaussian \( i \) has been visible.
- **Cloning Criteria**: Clone if: $\overline{\text{grad}}_i \geq \tau_{\text{clone}} \quad \text{and} \quad \max(\mathbf{s}_i) \leq \gamma_{\text{scale}} \cdot \text{scene\_extent}$
- **Splitting Criteria**: Split if: $\overline{\text{grad}}_i \geq \tau_{\text{split}} \quad \text{and} \quad \max(\mathbf{s}_i) > \gamma_{\text{scale}} \cdot \text{scene\_extent}$
- **Cloning Process**: Duplicate Gaussian $i$:  $\text{Create new Gaussian with parameters identical to Gaussian } i$
- **Splitting Process**: Generate \( N \) Gaussians from Gaussian \( i \):
    - New positions: $\mathbf{x}_{i, k, \text{new}} = \mathbf{x}_i + \mathbf{R}_i \Delta \mathbf{x}_{i, k}, \quad \Delta \mathbf{x}_{i, k} \sim \mathcal{N}(\mathbf{0}, \mathbf{S}_i)$
    - New scaling factors:  $\mathbf{s}_{i, \text{new}} = \ln\left( \frac{e^{\mathbf{s}_i}}{0.8 N} \right)$
    - Other parameters are duplicated.
- **Pruning Criteria**: Prune if: $\alpha_i < \tau_{\text{opacity}} \quad \text{or} \quad \text{size constraints}$

# getWorld2View2
#### **Function Purpose**
The `getWorld2View2` function computes a **world-to-camera transformation matrix**, which is essential in computer graphics and computer vision for mapping 3D points in the world coordinate system to the camera's view coordinate system.

#### **Inputs and Parameters**
1. **`R` (Rotation matrix)**: A $3 \times 3$ orthogonal matrix representing the orientation of the camera in the world. It satisfies $R R^T = I$ and $det(R) = 1$.
2. **`t` (Translation vector)**: A $3 \times 1$ vector representing the position of the camera in world coordinates.
3. **`translate` (Offset)**: A $3 \times 1$ vector for additional translation to adjust the camera center position.
4. **`scale`**: A scalar to uniformly scale the translation.

---

#### **Key Steps in the Function**
1. **Initialize the Transformation Matrix (`Rt`)**
    ```python
    Rt = np.zeros((4, 4))  # 4x4 matrix
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    ```

    Here, a $4 \times 4$ transformation matrix `Rt` is constructed. Its structure is: $Rt = \begin{bmatrix} R^T & t \\ 0 & 1 \end{bmatrix}$
    - $R^T$: The transpose of the rotation matrix $R$. This represents the inverse of rotation when applied to column vectors, which is needed for the transformation matrix.
    - $t$: The translation vector added as the last column to represent translation.
    - The last row $[0, 0, 0, 1]$ is used for homogeneous coordinates in 3D transformations.

---

2. **Compute Camera-to-World Matrix (`C2W`)**
    ```python
    C2W = np.linalg.inv(Rt)
    ```

Here, the camera-to-world matrix is computed as the inverse of the `Rt` matrix: $C2W = Rt^{-1}$

The `C2W` matrix transforms points from the camera's coordinate system back to the world coordinate system.

The inverse of a transformation matrix is given by:$C2W = \begin{bmatrix} R & -R t \\ 0 & 1 \end{bmatrix}^{-1} = \begin{bmatrix} R^T & R^T t \\ 0 & 1 \end{bmatrix}$

---

3. **Adjust the Camera Center**
    ```python
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    ```

    - The camera center is extracted from the $C2W$ matrix:$\text{cam\_center} = C2W[:3, 3]$This is the translation component of the matrix, indicating the position of the camera in world space.
    - This center is adjusted using the `translate` vector and scaled by `scale`:$\text{cam\_center} = (\text{cam\_center} + \text{translate}) \cdot \text{scale}$
    - The adjusted camera center is then reassigned to the `C2W` matrix.

---

4. **Recompute World-to-Camera Matrix**
    ```python
    Rt = np.linalg.inv(C2W)
    ```

The adjusted `C2W` matrix is inverted again to compute the final world-to-camera matrix: $Rt = C2W^{-1}$

---
#### **Output**
The function returns the final world-to-camera transformation matrix:
$Rt = \begin{bmatrix} R^T & t' \\ 0 & 1 \end{bmatrix}$
where $t'$ incorporates the effects of the adjusted camera center translation and scaling.

---
### Relevance to 3D Gaussian Splatting
In the context of 3D Gaussian Splatting:
1. **Transformation Matrices**: These are crucial for aligning 3D points or Gaussian distributions (represented in the world space) with the camera view. The transformations ensure that the 3D Gaussian centers are properly projected onto the 2D image plane.
2. **Scaling and Translation**: The ability to modify the camera center through translation and scaling can be used to experiment with different perspectives and alignments during training or rendering.

# load_image_camera_from_transforms
### **Function Purpose**
The `load_image_camera_from_transforms` function reads images and corresponding camera data from a `transforms.json` file, processes the data, and constructs a list of `Camera` objects. Each `Camera` object encapsulates image data, camera intrinsic and extrinsic parameters, and transformation matrices.

---
### **Class: Camera**
The `Camera` class is central to this function, as it encapsulates:
	1. **Intrinsic Parameters:** Field of View (FoV), near and far plane distances.
	2. **Extrinsic Parameters:** Rotation (`R`), translation (`t`), and transformation matrices.
	3. **Projection Matrix:** Converts 3D points into the 2D image plane.

---
#### **1. Projection Matrix Calculation**
The projection matrix `P` maps 3D coordinates in camera space to 2D image coordinates, incorporating perspective distortion.
```python
def getProjectionMatrix(znear, zfar, fovX, fovY):
    ...
```

Mathematical formulation of the projection matrix $P$: $P = \begin{bmatrix} \frac{2 z_{\text{near}}}{r - l} & 0 & \frac{r + l}{r - l} & 0 \\ 0 & \frac{2 z_{\text{near}}}{t - b} & \frac{t + b}{t - b} & 0 \\ 0 & 0 & \frac{z_{\text{far}}}{z_{\text{far}} - z_{\text{near}}} & -\frac{z_{\text{far}} z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}} \\ 0 & 0 & 1 & 0 \end{bmatrix}$

Where:
- $l, r, t, b$: Left, right, top, and bottom planes of the viewing frustum.
- $z_{\text{near}}, z_{\text{far}}$: Near and far plane distances.
- $\tan(\text{FoV})$: Determines $t, b, l, r$.

From the code:
$t = z_{\text{near}} \cdot \tan\left(\frac{\text{FoV}_Y}{2}\right), \quad b = -t $ $r = z_{\text{near}} \cdot \tan\left(\frac{\text{FoV}_X}{2}\right), \quad l = -r$

The diagonal terms of PP scale by $\frac{2 z_{\text{near}}}{r - l}$ and $\frac{2 z_{\text{near}}}{t - b}$, ensuring correct perspective projection.

---
#### **2. Extrinsic: World-to-Camera Transform**
The world-to-camera transform matrix is built using rotation $R$ and translation $t$. This matrix aligns the world coordinate system with the camera's frame:
$T_{\text{world-to-camera}} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$

Where:
- $R$: Camera's orientation (rotation matrix).
- $t$: Camera's position in the world frame.

The code processes this by:
```python
w2c = np.linalg.inv(c2w)
R, t = (np.transpose(w2c[:3, :3]), w2c[:3, 3])
```

Here, `c2w` is the camera-to-world transformation from the `transforms.json` file. Its inverse gives $T_{\text{world-to-camera}}$.

---
#### **3. Intrinsic: Field of View to Focal Length**
Conversion between FoV and focal length is derived from the pinhole camera model:
$f = \frac{p}{2 \cdot \tan\left(\frac{\text{FoV}}{2}\right)}$
Where:
- $f$: Focal length.
- $p$: Image width in pixels.

The code implements this as:
```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
```

---
#### **4. World View and Full Projection Transformation**
The full transformation from world space to image space is computed by combining:
1. The world-to-view transform (`world_view_transform`).
2. The projection matrix (`getProjectionMatrix`).

This is represented as:
$P_{\text{full}} = T_{\text{world-to-camera}} \cdot P$

In code:
```python
self.full_proj_transform = (
    self.world_view_transform.unsqueeze(0).bmm(
        getProjectionMatrix(...).transpose(0, 1).to(device).unsqueeze(0)
    )
).squeeze(0)
```

---
### **Processing the Image Data**
The image data is loaded and processed to:
1. Normalize pixel values to $[0, 1]$.
2. Apply background blending based on alpha channel.
3. Reshape into a format suitable for GPU processing.

```python
image_norm = np.array(Image.open(image_path).convert("RGBA")) / 255.0
image_back = ...
image_fore = ...
image_data = Image.fromarray(image_fore + image_back, "RGB").resize(resolution)
```

Blending formula:
$background\text{blended\_pixel} = \text{alpha} \cdot \text{foreground} + (1 - \text{alpha}) \cdot \text{background}$

---
### **Main Function Workflow**
1. **Load `transforms.json`**:
    - Extract camera extrinsic (`c2w`) and field of view (`fovx`).
2. **Process Images**:
    - Normalize, blend, and resize.
3. **Compute Camera Parameters**:
    - Rotation RR, translation tt, and FoV-to-focal conversions.
4. **Initialize `Camera` Objects**:
    - Create and store each camera’s intrinsic and extrinsic matrices.
---
### **Mathematical Connection to the Paper**
This code likely corresponds to preprocessing steps in the "3D Gaussian Splatting" paper, where camera matrices are vital for rendering or scene reconstruction tasks. Specifically:
- **Projection Matrices**: Enable mapping between 3D Gaussian primitives and the 2D image plane.
- **Camera Transformations**: Align world space representations with image observations.

# getNerfppNorm
### **Overview**

This code defines a function `getNerfppNorm` that normalizes camera positions for consistent scaling in a 3D scene, based on the **NeRF++ (Neural Radiance Fields++)** approach. It computes the scene's bounding sphere by calculating the centroid of all camera positions and the radius that bounds them.

The computed normalization is then applied in the `GsDataset` class using `load_image_camera_from_transforms` to load camera data and set `cameras_extent` for the dataset.

---

### **Detailed Breakdown**
#### **1. `getNerfppNorm` Function**
This function normalizes camera positions to align and scale the scene consistently.
##### **Step 1.1: `get_center_and_diag`**
The helper function `get_center_and_diag` calculates the **centroid** and **diagonal radius** of all camera centers.
1. **Camera Center Aggregation:** The input `cam_centers` is an $N \times 3$ matrix, where each row represents a 3D camera center $\mathbf{c}_i$.
    In the code:
    ```python
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    ```
    - $\mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_N] \in \mathbb{R}^{3 \times N}$
    - Centroid center: $\mathbf{center} = \frac{1}{N} \sum_{i=1}^N \mathbf{c}_i$
2. **Diagonal Computation:** The diagonal is defined as the largest distance between any camera center and the centroid. The distance for each center is: $\text{dist}_i = \|\mathbf{c}_i - \mathbf{center}\|_2$The diagonal is:$\text{diagonal} = \max(\text{dist}_1, \text{dist}_2, \dots, \text{dist}_N)$In the code:
    ```python
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    ```

##### **Step 1.2: Extract Camera Centers**
The camera centers are extracted by inverting the **world-to-camera transform** $W2C$, then reading the translation component $\mathbf{t}_{\text{cam}}$:
$\text{Camera Center} = C2W[:3, 3]$

This is implemented in the loop:
```python
for cam in cam_info:
    W2C = getWorld2View2(cam.R, cam.t)
    C2W = np.linalg.inv(W2C)
    cam_centers.append(C2W[:3, 3:4])
```

##### **Step 1.3: Compute Scene Normalization**
The normalization consists of:
1. **Translation**: Negating the centroid $\mathbf{center}$ aligns the scene to the origin.$\text{translate} = -\mathbf{center}$
2. **Radius**: Enlarging the diagonal by 10% ensures all cameras fit within a bounding sphere: $\text{radius} = 1.1 \cdot \text{diagonal}$
The final result is:
```python
return {"translate": translate, "radius": radius}
```

---
#### **2. Usage in `GsDataset`**
The normalization is applied to the loaded camera data:
```python
self.image_camera = load_image_camera_from_transforms(device, image_camera_path, resolution)
self.cameras_extent = getNerfppNorm(self.image_camera)["radius"]
```
1. **Load Camera Data**: The `load_image_camera_from_transforms` function returns a list of `Camera` objects, which include rotation matrices (RR), translations (t\mathbf{t}), and other camera parameters.
2. **Normalize Scene**: The normalization ensures the cameras and the scene are scaled and positioned consistently, with the `radius` representing the scene's extent.
---
### **Mathematical Connection to NeRF++**
In **NeRF++**, camera normalization helps:
- Align the scene to the origin for computational efficiency.
- Scale the scene to fit within a unit sphere for consistent neural network input.

The radius $\text{radius}$ provides a consistent reference for scene extents, ensuring uniform sampling and training stability across different datasets.

This normalization process connects with preprocessing requirements in 3D rendering pipelines, particularly for methods like NeRF++, where consistent spatial scaling and alignment are critical. Let me know if anything needs clarification or further elaboration!

# GsNetwork init
Let’s delve into the **`GsNetwork`** class thoroughly, covering **positions**, **covariances**, **colors**, and **opacities**, step-by-step. Each component will be tied to its mathematical formula and explained in detail.

---
### **1. Class Initialization**
The `__init__` method initializes all components of the network related to the 3D Gaussian representation.
#### **1.1 Input Parameters**
- **`device`**: Specifies the computation device (CPU/GPU).
- **`point_number`**: Total number of 3D Gaussians (or points) being initialized.
- **`percent_dense`**: Controls density adjustments (used in further computations, not directly part of the formulas here).
- **`max_sh_degree`**: Maximum degree of spherical harmonics (SH), influencing the representation's view-dependence and feature capacity.
---
### **2. Position Initialization**
#### Code:
```python
points = (torch.rand(point_number, 3).float().to(device) - 0.5) * 1.0
self._xyz = nn.Parameter(points.requires_grad_(True))
```
#### Explanation:
- **Initialization**: `points` are sampled randomly from a uniform distribution in the range $[-0.5, 0.5]$.
- Each 3D Gaussian has a **position** $\mathbf{c}_i = (x_i, y_i, z_i)$, representing its center in 3D space.
- **Formula** $\mathbf{c}_i \sim \mathcal{U}([-0.5, 0.5]^3), \quad i = 1, \dots, \texttt{point\_number}.$
- **Parameter**: The positions are stored in `self._xyz` as a learnable tensor, enabling optimization during training.
---
### **3. Covariance Matrix Initialization**
Covariance matrices determine the size, orientation, and shape of the Gaussian blobs.
#### **3.1 Scale Initialization**
#### Code:
```python
scale = torch.log(torch.sqrt(torch.clamp_min(distCUDA2(points).float(), 0.0000001)))[..., None].repeat(1, 3)
self._scaling = nn.Parameter(scale.requires_grad_(True))
```
#### Explanation:
- `scale` represents the diagonal components of the scaling matrix $S = \text{diag}(\mathbf{s})$, which controls the axis-aligned size of each Gaussian.
- Distances between points are calculated using `distCUDA2`, and values are log-transformed for numerical stability.
- **Formula**: $\mathbf{s}_i = \log\left(\sqrt{\max(d_i, \epsilon)}\right),$
    where:
    - $d_i$: Distance of the $i-th$ point from its nearest neighbors (via `distCUDA2`).
    - $\epsilon$: Small constant ($10^{-7}$) for numerical stability.
---
#### **3.2 Rotation Initialization**
#### Code:
```python
rotation = torch.cat((torch.ones((point_number, 1)).float().to(device),
                      torch.zeros((point_number, 3)).float().to(device)), dim=1)
self._rotation = nn.Parameter(rotation.requires_grad_(True))
```
#### Explanation:
- Each rotation is represented as a **quaternion** $[r, x, y, z]$, initialized to $[1, 0, 0, 0]$ (identity quaternion).
- **Rotation Matrix**: A quaternion is converted into a $3 \times 3$ rotation matrix $R$: $R = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - rz) & 2(xz + ry) \\ 2(xy + rz) & 1 - 2(x^2 + z^2) & 2(yz - rx) \\ 2(xz - ry) & 2(yz + rx) & 1 - 2(x^2 + y^2) \end{bmatrix}$
- The rotation matrix $R$ is later combined with the scaling matrix SS to produce the covariance matrix.
---
#### **3.3 Covariance Matrix Construction**
The covariance matrix $\Sigma$ combines scaling and rotation.
#### Code:
```python
L = build_scaling_rotation(scaling_modifier * scaling, rotation)
actual_covariance = L @ L.transpose(1, 2)
```
#### Explanation:
- $L$ is the Cholesky decomposition of $\Sigma$, combining:
    - Diagonal scaling $S = \text{diag}(\mathbf{s})$.
    - Rotation matrix $R$ derived from the quaternion.
- **Covariance Formula**: $\Sigma = L L^\top, \quad L = R \cdot S.$
- Compact Representation: The independent entries of $\Sigma$ (for storage): $[\sigma_{11}, \sigma_{12}, \sigma_{13}, \sigma_{22}, \sigma_{23}, \sigma_{33}].$
---
### **4. Feature Initialization (Colors)**
#### Code:
```python
features = torch.cat((torch.rand(point_number, 3, 1).float().to(device) / 5.0 + 0.4,
                      torch.zeros((point_number, 3, (self.max_sh_degree + 1) ** 2 -1)).float().to(device)), dim=-1)
self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
```
#### Explanation:
- The features represent RGB values and spherical harmonic (SH) coefficients:
    - First 3 channels: RGB intensities, initialized to random values $\in [0.4, 0.6]$.
    - Remaining channels: SH coefficients for view-dependent effects, initialized to 0.
- **Formula**:
    - RGB: $\mathbf{f}_{\text{RGB}, i} \sim \mathcal{U}([0.4, 0.6])$.
    - SH Coefficients: $\mathbf{f}_{\text{SH}, i} = [0, \dots, 0], \quad \text{for } l = 1, \dots, \texttt{max\_sh\_degree}.$
---
### **5. Opacity Initialization**
#### Code:
```python
opacity = torch.log((torch.ones((point_number, 1)).float().to(device) * 0.1) /
                    (1. - (torch.ones((point_number, 1)).float().to(device) * 0.1)))
self._opacity = nn.Parameter(opacity.requires_grad_(True))
```
#### Explanation:
- **Opacity** ($\alpha_i$) controls the transparency of each Gaussian, stored in its **log-sigmoid** inverse form for numerical stability: $\text{sigmoid-inverse}(\alpha) = \log\left(\frac{\alpha}{1 - \alpha}\right)$
- **Initialization**: Opacity is initialized to 0.1: $\alpha_i = 0.1, \quad \sigma^{-1}(\alpha_i) = \log\left(\frac{0.1}{1 - 0.1}\right)$
---
### **6. Activations**
Activations ensure parameters are transformed appropriately during training:
- **Scaling**: Exponential to ensure positive values: $\mathbf{s} = \exp(\text{log-scaling}).$
- **Rotation**: Quaternions are normalized: $\mathbf{q} = \frac{\mathbf{q}}{\|\mathbf{q}\|}.$
- **Opacity**: Sigmoid function ensures values in [0,1][0, 1]: $\alpha = \sigma(x) = \frac{1}{1 + \exp(-x)}.$
---

# rasterization forward
The code you've provided is part of the GPU implementation for rendering 3D Gaussian splats as described in the paper **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** by **Karnovets et al.** This implementation focuses on efficiently rendering point clouds represented as oriented 3D Gaussians using GPU rasterization techniques.

Below, I'll explain the code step by step, providing mathematical formulations and connecting them to relevant sections of the paper.

---
## Overview
The rendering pipeline involves the following main steps:
1. **Compute 3D Covariance Matrices**: For each Gaussian, compute its 3D covariance matrix based on scale and rotation.
2. **Project to 2D Covariance Matrices**: Project the 3D covariances into 2D screen space to determine the splat's shape on the screen.
3. **Compute Colors from Spherical Harmonics**: Evaluate spherical harmonics (SH) to obtain colors for each Gaussian based on view direction.
4. **Preprocess Gaussians**: Perform initial computations for each Gaussian before rasterization.
5. **Rasterize Gaussians**: Render the Gaussians onto the screen using GPU rasterization.
---
## 1. Compute Color from Spherical Harmonics (`computeColorFromSH`)
### Purpose:
Compute the RGB color of each Gaussian by evaluating spherical harmonics (SH) coefficients in the view direction.
### Mathematical Explanation:
The color \( c \) of a Gaussian is computed using spherical harmonics evaluated at the direction from the camera to the Gaussian. The SH expansion up to degree \( l \) can be written as:
$$c(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} y_{lm}(\theta, \phi) \cdot c_{lm}$$
Where:
- $y_{lm}(\theta, \phi)$ are the real SH basis functions.
- $c_{lm}$ are the SH coefficients for RGB channels.

In the code, the SH basis functions are evaluated up to a certain degree (e.g., degree 2 or 3).

### Code
- **Compute View Direction**: The direction vector `dir` is computed from the Gaussian position `pos` to the camera position `campos`, and then normalized.
```cpp
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);
```

- **Access SH Coefficients**: The SH coefficients `sh` for the current Gaussian are retrieved.
```cpp
  glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
```

- **Evaluate SH Basis Functions**: The SH basis functions are evaluated at the direction `dir`. Constants like `SH_C0`, `SH_C1`, `SH_C2`, etc., represent precomputed normalization factors for SH basis functions.
```cpp
  glm::vec3 result = SH_C0 * sh[0];
  // For degree > 0, add higher-order terms
```

- **Clamping and Shifting**: After computing the color, 0.5 is added to shift the color range, and the result is clamped to ensure non-negative RGB values.
```cpp
  result += 0.5f;
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
```

**Section 3.4 (Radiance Field Representation)**: The paper describes using SH coefficients to represent view-dependent color variations.

---
## 2. Compute 3D Covariance Matrices (`computeCov3D`)
### Purpose:
Compute the 3D covariance matrix $\Sigma$ for each Gaussian based on its scale and rotation parameters.
### Mathematical Explanation:
The covariance matrix $\Sigma$ of a 3D oriented Gaussian is given by:
$$\Sigma = (S R)^T (S R) = R^T S^T S R$$
Where:
- $S$ is the diagonal scale matrix.  $S = \begin{bmatrix}  s_x & 0 & 0 \\  0 & s_y & 0 \\  0 & 0 & s_z \\  \end{bmatrix}$
- $R$ is the rotation matrix derived from the quaternion.
### Code:
- **Create Scale Matrix** **$S$**:
```cpp
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;
```

- **Normalize Quaternion and Compute Rotation Matrix** $R$:
```cpp
  // Normalize quaternion q = (r, x, y, z)
  glm::vec4 q = rot; // Assuming rot is already normalized
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;
 
  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(
    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
  );
```

- **Compute** $\Sigma = (S R)^T (S R)$:
```cpp
  glm::mat3 M = S * R;
  glm::mat3 Sigma = glm::transpose(M) * M;
```

- **Store Upper Triangular Part**: Since $\Sigma$ is symmetric, only the upper triangular part is stored to save memory.
```cpp
  cov3D[0] = Sigma[0][0]; // \Sigma_{xx}
  cov3D[1] = Sigma[0][1]; // \Sigma_{xy}
  cov3D[2] = Sigma[0][2]; // \Sigma_{xz}
  cov3D[3] = Sigma[1][1]; // \Sigma_{yy}
  cov3D[4] = Sigma[1][2]; // \Sigma_{yz}
  cov3D[5] = Sigma[2][2]; // \Sigma_{zz}
```

- **Section 3.2 (Gaussian Representation)**: The paper defines each Gaussian by its mean $\mu$ and covariance $\Sigma$, capturing the orientation and spatial extent of the Gaussian in 3D space.
---
## 3. Project to 2D Covariance Matrices (`computeCov2D`)
### Purpose:
Project the 3D covariance matrix $\Sigma$ to a 2D screen-space covariance matrix $\Sigma_{\text{scr}}$, which determines the shape and size of the Gaussian splat on the screen.
### Mathematical Explanation:
The projection involves several steps, following equations from the paper:
1. **Compute Jacobian \( J \)** of the projection function.  $$J = \begin{bmatrix} f_x / z & 0 & -f_x x / z^2 \\ 0 & f_y / z & -f_y y / z^2 \\ 0 & 0 & 0 \\ \end{bmatrix}$$
   Where:
   - $f_x, f_y$ are focal lengths along x and y axes.
   - $x, y, z$ are coordinates of the point in camera space.
2. **Compute Transformation Matrix $T = W J$**, where $W$ is the world-to-view matrix.
3. **Compute Screen-Space Covariance**:  $$\Sigma_{\text{scr}} = T^T \Sigma T$$
### Code:
- **Transform Point to View Space**:
```cpp
  float3 t = transformPoint4x3(mean, viewmatrix);
```

- **Limit Clipping**: Clip the projected coordinates to avoid distortion at large angles.
```cpp
  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  // Clipping logic
```

- **Compute Jacobian \( J \)**:
```cpp
  glm::mat3 J = glm::mat3(
    focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
    0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0, 0, 0);
```

- **Compute Transformation Matrix \( T = W J \)**:
```cpp
  glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[4], viewmatrix[8],
    viewmatrix[1], viewmatrix[5], viewmatrix[9],
    viewmatrix[2], viewmatrix[6], viewmatrix[10]);
  glm::mat3 T = W * J;
```

- **Recover Original Covariance Matrix \( $\Sigma$ \) from Stored Values**:
```cpp
  glm::mat3 Vrk = glm::mat3(
    cov3D[0], cov3D[1], cov3D[2],
    cov3D[1], cov3D[3], cov3D[4],
    cov3D[2], cov3D[4], cov3D[5]);
```

- **Compute Screen-Space Covariance $\Sigma_{\text{scr}}$**:
```cpp
  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
```

- **Add Low-Pass Filter**: Ensure that each Gaussian covers at least one pixel to prevent aliasing.
```cpp
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
```

- **Return Upper Triangular Part**:
```cpp
  return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
```

**Section 4.1 (Screen Space Projection)**: The projection of Gaussians into screen space is described, including the computation of the screen-space covariance matrix.

---
## 4. Preprocess Gaussians (`preprocessCUDA`)
### Purpose:
Perform initial computations for each Gaussian before rasterization, including culling, projection, and preparation of data needed for rendering.
### Mathematical Explanation:
This function incorporates several steps:
- **Frustum Culling**: Determine if the Gaussian is within the camera's view frustum.
- **Projection**: Project Gaussians onto the screen.
- **Compute Screen-space Covariances**: Use `computeCov2D` to get the 2D covariances.
- **Compute Colors**: Use `computeColorFromSH` to get the RGB colors.
- **Determine Splat Size**: Calculate the extent (radius) of the splat on the screen.
- **Prepare Binning Data**: Determine which tiles (screen regions) the splat overlaps.

### Code:
- **Initialize Variables and Cull Gaussians Outside the Frustum**:
```cpp
  int idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;
 
  // Initialize radius and tiles touched
  radii[idx] = 0;
  tiles_touched[idx] = 0;
 
  // Near culling
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
    return;
```

- **Project Points**:
```cpp
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.00001f);
  float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
```

- **Compute or Retrieve 3D Covariance**:
```cpp
  if (cov3D_precomp != nullptr)
    cov3D = cov3D_precomp + idx * 6;
  else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }
```

- **Compute 2D Covariance**:
```cpp
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
```

- **Compute Inverse Covariance (For EWA Splatting)**:
```cpp
  float det = (cov.x * cov.z - cov.y * cov.y);
  float det_inv = 1.f / det;
  float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
```

- **Compute Splat Radius and Clipping Rectangle**:
```cpp
  // Compute eigenvalues (lambda1, lambda2) of the covariance matrix
  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
 
  // Compute pixel position
  float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
 
  // Determine overlapping tiles
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
```

 The multiplication by 3 corresponds to $3\sigma$ of the Gaussian, covering 99.7% of the distribution.

- **Compute Colors**:
```cpp
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }
```

- **Store Helper Data for Rasterization**:
```cpp
  depths[idx] = p_view.z;
  radii[idx] = my_radius;
  points_xy_image[idx] = point_image;
  conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```

**Section 4 (Efficient Rendering)**: Discusses efficient culling, projection, and preparation of Gaussians for rasterization.

---
## 5. Rasterize Gaussians (`renderCUDA`)
### Purpose:
Render the Gaussians onto the screen using GPU rasterization, applying blending and accumulation to produce the final image.

### Mathematical Explanation:
The rendering uses an **Order-Independent Transparency** approach, blending Gaussians based on their alpha values without sorting.
- **Alpha Blending**: The color \( C \) and alpha \( T \) are updated per pixel:
  $$C_{\text{out}} = C_{\text{in}} + \alpha \cdot T_{\text{in}}  $T_{\text{out}} = T_{\text{in}} \cdot (1 - \alpha)$$
- **Gaussian Weight**: The alpha \( \alpha \) for each Gaussian is computed based on its opacity and the Gaussian function evaluated at the pixel position: $$\alpha = \min(0.99, \text{opacity} \times e^{-0.5 \cdot d^T \Sigma^{-1} d})$$
  Where $d$ is the distance from the Gaussian center to the pixel, and $\Sigma^{-1}$ is the inverse covariance matrix.

### Code:
- **Set Up Pixel Coordinates**:
```cpp
  uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
  float2 pixf = { (float)pix.x, (float)pix.y };
```

- **Initialize Blending Variables**:
```cpp
  float T = 1.0f; // Remaining transparency
  float C[CHANNELS] = { 0 }; // Accumulated color
```

- **Iterate Over Gaussians in the Tile**:
```cpp
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // Fetch Gaussians and shared data
    // ...
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Compute Gaussian weight (alpha)
      float2 d = { xy.x - pixf.x, xy.y - pixf.y };
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;
     
      float alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < 1.0f / 255.0f)
        continue;
     
      // Update color and transparency
      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }
      for (int ch = 0; ch < CHANNELS; ch++)
        C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
      T = test_T;
    }
  }
```

- **Write Final Colors to Output**:
```cpp
  for (int ch = 0; ch < CHANNELS; ch++)
    out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
```

**Section 4.2 (Rasterization and Blending)**: Describes the rasterization process, including how Gaussians are accumulated using order-independent transparency.

---
## Additional Notes
- **EWA Splatting**: The code refers to **EWA (Elliptical Weighted Average) Splatting**, a technique used for high-quality texture filtering and rendering of splats. The method involves computing an anisotropic kernel (here, a Gaussian) in screen space for each splat. This is connected to the computation of inverse covariance matrices and blending weights.
- **Screen-Space Extent**: The radius computation using eigenvalues ensures that the splat is rendered over an area that covers most of its contribution.
- **Efficient GPU Implementation**: The code uses shared memory and cooperative groups to optimize memory access and parallel computation on the GPU.
- **Tile-Based Rasterization**: The rendering is performed per screen tile (block), which improves cache coherence and memory access patterns.

# rasterization backward
## Overview
The backward pass is essential for training neural networks using gradient-based optimization. It involves computing the gradients of the loss function with respect to the model's parameters. In the context of 3D Gaussian Splatting, these parameters include:
- **Mean positions** of the Gaussians.
- **Scales and rotations** (defining the 3D covariance matrices).
- **Spherical Harmonics (SH) coefficients** (defining the color of the Gaussians).
- **Opacities** of the Gaussians.

The backward pass computes how changes in these parameters affect the loss, enabling the optimization process.

The **backward.cu** file contains several functions that perform the backward computation:
1. `computeColorFromSH`: Backpropagated through the color computation from SH coefficients.
2. `computeCov2DCUDA`: Backpropagated through the projection and inversion of 2D covariance matrices.
3. `computeCov3D`: Backpropagated through the computation of 3D covariance matrices from scales and rotations.
4. `preprocessCUDA`: Aggregates gradients from the backward passes.
5. `renderCUDA`: Backpropagated through the rendering step, computing gradients w.r.t. per-pixel values.
6. `BACKWARD::preprocess`: Wrapper function for preprocessing backward passes.
7. `BACKWARD::render`: Wrapper function for rendering backward passes.

Let's explain each of these functions in detail.

---
## 1. Backward Pass for Color Computation from SH (`computeColorFromSH`)
### Purpose:
Compute gradients of the loss with respect to:
- **SH coefficients**: $\partial \text{SH}$
- **Gaussian mean positions**: $\frac{\partial L}{\partial \mu}$
### Mathematical Explanation:
In the forward pass, the RGB color $c$ is computed using SH coefficients evaluated at the view direction $mathbf{d}$:
$$c = \sum_{l=0}^{L} \sum_{m=-l}^{l} y_{lm}(\mathbf{d}) \cdot \text{SH}_{lm}  $$
Where:
- $y_{lm}(\mathbf{d})$ are the SH basis functions.
- $\text{SH}_{lm}$ are the SH coefficients for the Gaussian.

In the backward pass, given the gradient of the loss w.r.t. the color $\frac{\partial L}{\partial c}$, we need to compute gradients w.r.t.:
- SH coefficients $\frac{\partial L}{\partial \text{SH}_{lm}}$
- Gaussian mean positions $\frac{\partial L}{\partial \mu}$
### Connection to the Code:
- **Compute View Direction**:
```cpp
  glm::vec3 pos = means[idx];
  glm::vec3 dir_orig = pos - campos;
  glm::vec3 dir = dir_orig / glm::length(dir_orig);
```

- **Retrieve SH Coefficients**:
```cpp
  glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
```

- **Get Gradient of Loss w.r.t. Color**:
```cpp
  glm::vec3 dL_dRGB = dL_dcolor[idx];
```

- **Clamping Gradients (if applicable)**: If clamping was applied during the forward pass (e.g., to ensure non-negative colors), gradients are adjusted accordingly.
```cpp
  dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
  dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
  dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
```

- **Compute Gradients w.r.t. SH Coefficients**:
  For each SH coefficient, compute:
$$
\frac{\partial L}{\partial \text{SH}_{lm}} = y_{lm}(\mathbf{d}) \cdot \frac{\partial L}{\partial c}
$$
 
  For example, for degree 0:
```cpp
  float dRGBdsh0 = SH_C0;
  dL_dsh[0] = dRGBdsh0 * dL_dRGB;
```

- **Compute Gradients w.r.t. Direction**:
  Since SH basis functions depend on the direction $\mathbf{d}$, we compute: $$\frac{\partial L}{\partial \mathbf{d}} = \sum_{lm} \frac{\partial y_{lm}}{\partial \mathbf{d}} \cdot \text{SH}_{lm} \cdot \frac{\partial L}{\partial c}$$
  The code computes the partial derivatives of SH basis functions w.r.t. x, y, z components of the direction and accumulates them in `dRGBdx`, `dRGBdy`, `dRGBdz`.

- **Compute Gradients w.r.t. Gaussian Mean Positions**:
  The view direction $\mathbf{d}$ depends on the Gaussian's position $\mu$: $$\mathbf{d} = \frac{\mu - \text{campos}}{\|\mu - \text{campos}\|}$$
 
  Using the chain rule: $$\frac{\partial L}{\partial \mu} = \left( \frac{\partial \mathbf{d}}{\partial \mu} \right)^T \frac{\partial L}{\partial \mathbf{d}}$$

 The code adjusts for the normalization of $\mathbf{d}$ by computing:
```cpp
  glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
  float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);
```

  Here, `dnormvdv` computes the derivative of a normalized vector w.r.t. the original vector.

- **Accumulate Gradient w.r.t. Mean**:
```cpp
  dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
```

**Section 3.4 (Radiance Field Representation)**: Discusses the use of SH coefficients to represent color variations, and optimizing them requires computing gradients as shown.

---
## 2. Backward Pass for 2D Covariance Matrix Inversion (`computeCov2DCUDA`)

### Purpose:
Compute gradients of the loss with respect to:
- **3D covariance matrices**: $\frac{\partial L}{\partial \Sigma}$
- **Gaussian means**: $\frac{\partial L}{\partial \mu}$

By backpropagating through the projection of 3D covariances into screen space and the inversion of the 2D covariances.
### Mathematical Explanation:
In the forward pass, after projecting to screen space, the 2D covariance matrix is inverted to get the conic parameters used in the rendering.

Given that:
- $\Sigma_{\text{scr}} = T^T \Sigma T$
- $\Sigma_{\text{scr}}$ is a 2x2 matrix.
- The inverse of $\Sigma_{\text{scr}}$ is used in the Gaussian evaluation.

In the backward pass, we have the gradient of the loss w.r.t. the conic parameters (inverse covariance elements):
$$
\frac{\partial L}{\partial \Sigma_{\text{scr}}^{-1}}
$$

We need to compute: $\frac{\partial L}{\partial \Sigma_{\text{scr}}}$, $\frac{\partial L}{\partial T}$, $\frac{\partial L}{\partial \Sigma}$, $\frac{\partial L}{\partial \mu}$
### Code:
**Retrieve Necessary Variables**:
  - The 3D covariance (`cov3D`).
  - The transformed mean position (`t`).
  - Gradient w.r.t. the conic parameters (`dL_dconics`).

**Compute Clipping Factors**:  Adjust gradients for any clipping that occurred in the forward pass, to avoid invalid values.

**Recompute Jacobian \( J \) and \( T \)**:  As in the forward pass, compute:
```cpp
  glm::mat3 J = ...;
  glm::mat3 W = ...;
  glm::mat3 T = W * J;
```

**Compute Screen-Space Covariance $\Sigma_{\text{scr}}$**:
```cpp
  glm::mat3 Vrk = ...; // 3D covariance matrix
  glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
```

**Add Low-Pass Filter Term**:
```cpp
  float a = cov2D[0][0] += 0.3f;
  float b = cov2D[0][1];
  float c = cov2D[1][1] += 0.3f;
```

**Compute Determinant and Its Inverse**:
```cpp
  float denom = a * c - b * b;
  float denom2inv = 1.0f / ((denom * denom) + 0.00001f);
```

**Compute Gradients w.r.t. Covariance Elements**: Using the expression for the inverse of a 2x2 matrix and its derivative, compute:
```cpp
  dL_da = ...;
  dL_db = ...;
  dL_dc = ...;
```

**Compute Gradients w.r.t. 3D Covariance $\Sigma$**:
```cpp
  // Compute gradients w.r.t. 3D covariance elements
  dL_dcov[6 * idx + 0] = ...; // \Sigma_{xx}
  dL_dcov[6 * idx + 1] = ...; // \Sigma_{xy}
  // and so on...
```

**Compute Gradients w.r.t. Transformation Matrix \( T \)**:
```cpp
  // Compute dL_dT00, dL_dT01, ..., dL_dT12
```

**Compute Gradients w.r.t. Jacobian \( J \)**:  Since $T = W J$, we have:
```cpp
  float dL_dJ00 = ...;
  float dL_dJ02 = ...;
  float dL_dJ11 = ...;
  float dL_dJ12 = ...;
```

**Compute Gradients w.r.t. Transformed Mean \( t \)**:
```cpp
  float dL_dtx = ...;
  float dL_dty = ...;
  float dL_dtz = ...;
```

**Compute Gradients w.r.t. Gaussian Mean \( $\mu$ \)**:  Transform the gradients back to the original coordinate system:
```cpp
  float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);
```

**Accumulate Gradients w.r.t. Mean**:
```cpp
  dL_dmeans[idx] = dL_dmean;
```

**Section 4.1 (Screen Space Projection)**: The backward computation closely follows the forward steps of projecting and inverting the covariance matrices.

---
## 3. Backward Pass for 3D Covariance Computation (`computeCov3D`)
### Purpose:
Compute gradients of the loss with respect to:
- **Scales**: $\frac{\partial L}{\partial s}$
- **Rotations**: $\frac{\partial L}{\partial q}$

By backpropagating through the computation of the 3D covariance matrix from scales and rotations.
### Mathematical Explanation:
Given:
- $\Sigma = (S R)^T (S R) = R^T S^T S R$
- $M = S R$

The gradient w.r.t. $M$ is: $$\frac{\partial L}{\partial M} = 2 M \frac{\partial L}{\partial \Sigma}$$

Then we compute gradients w.r.t.:
- Scales \( s \):
$$
 \frac{\partial L}{\partial s_i} = \text{row}_i(R^T) \cdot \text{column}_i\left( \frac{\partial L}{\partial M^T} \right)
$$

- Rotations \( R \): $$\frac{\partial L}{\partial R} = S^T \frac{\partial L}{\partial M}$$And since $R$ is a function of quaternion $q$ , we need to compute $\frac{\partial L}{\partial q}$.
### Code:
**Recompute Rotation Matrix \( R \)**:
```cpp
  glm::vec4 q = rot;
  float r = q.x, x = q.y, y = q.z, z = q.w;
  glm::mat3 R = ...; // Recompute R from q
```

**Recompute Scale Matrix \( S \)**:
```cpp
  glm::mat3 S = glm::mat3(1.0f);
  glm::vec3 s = mod * scale;
  S[0][0] = s.x;
  S[1][1] = s.y;
  S[2][2] = s.z;
```

**Compute $M = S R$**:
```cpp
  glm::mat3 M = S * R;
```

**Compute Gradient w.r.t. $M$**:
```cpp
  glm::mat3 dL_dSigma; // Constructed from dL_dcov3Ds
  glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
```

**Compute Gradient w.r.t. Scales**:
```cpp
  glm::mat3 Rt = glm::transpose(R);
  glm::mat3 dL_dMt = glm::transpose(dL_dM);
  dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
  dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
  dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
```

**Compute Gradient w.r.t. Rotations**:  Compute derivative of $M$ w.r.t. quaternion $q$, then compute $\frac{\partial L}{\partial q}$:
```cpp
  // Compute dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w
```

**Normalize Quaternion Gradient** (if necessary):  Since quaternions are often normalized to represent rotations, the gradient needs to account for this normalization.
```cpp
  // Apply normalization to dL_drot
```

**Section 3.2 (Gaussian Representation)**: The representation of Gaussians using scales and rotations is critical, and optimizing them requires backpropagating through these computations.

---
## 4. Backward Pass for Preprocessing (`preprocessCUDA`)
### Purpose:
Aggregate gradients from previous backward functions and propagate them to the initial Gaussian parameters.

### Connection to the Code:
**Compute Gradient w.r.t. Mean**:  Combine gradients from:
  - The projection of the mean to 2D (`dL_dmean2D`).
  - The SH to RGB conversion (via `computeColorFromSH`).
  - The covariance computation (via `computeCov3D` and `computeCov2DCUDA`).
```cpp
  glm::vec3 dL_dmean;
  // Compute dL_dmean using projmatrix and dL_dmean2D
  dL_dmeans[idx] += dL_dmean;
```

**Compute Gradient w.r.t. SH Coefficients**:  If SH coefficients are used, call `computeColorFromSH` to compute the gradients.
```cpp
  if (shs)
      computeColorFromSH(...);
```

**Compute Gradient w.r.t. Scales and Rotations**:  Use `computeCov3D` to compute the gradients.
```cpp
  if (scales)
      computeCov3D(...);
```

The aggregation of gradients is necessary for updating Gaussian parameters during optimization.

---
## 5. Backward Pass for Rendering (`renderCUDA`)
### Purpose:
Compute gradients of the loss with respect to:
- **Gaussian colors**
- **Opacities**
- **2D positions and covariances**

By backpropagating through the rendering pipeline.
### Mathematical Explanation:
In the forward rendering, the pixel color is computed as:
$$
C_{\text{pixel}} = \left( \prod_{i} (1 - \alpha_i) \right) C_{\text{bg}} + \sum_{i} \left( \alpha_i \prod_{j < i} (1 - \alpha_j) \right) c_i
$$

In the backward pass:
- Compute $\frac{\partial L}{\partial c_i}$ for each Gaussian contributing to the pixel.
- Propagate gradients w.r.t. $\alpha_i$, which depends on opacity and Gaussian evaluation at the pixel.
- Compute gradients w.r.t. 2D position and conic parameters.
### Connection to the Code:
**Initialize Variables**:
```cpp
  float T = T_final; // From the forward pass
  float accum_rec[C] = { 0 }; // Accumulated color
```

**Traverse Gaussians in Reverse Order**:
```cpp
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
  {
      // Load data into shared memory
      // Iterate over Gaussians
  }
```

**Compute Gradients**:  For each Gaussian:
  - Compute the Gaussian weight \( G \) at the pixel.
  - Compute alpha $\alpha = \min(0.99, \text{opacity} \times G)$.
  - Update transparency $T$.
  - Compute the gradient w.r.t. color $\frac{\partial L}{\partial c_i}$.
  - Compute the gradient w.r.t. alpha $\frac{\partial L}{\partial \alpha_i}$.
  - Compute the gradient w.r.t. opacity $\frac{\partial L}{\partial \text{opacity}_i}$.
  - Compute the gradient w.r.t. 2D mean position and conic parameters.
 
Atomic operations (`atomicAdd`) are used to safely accumulate gradients for shared Gaussians.

**Section 4.2 (Rasterization and Blending)**: The backward pass accurately mirrors the forward rendering to enable gradient computation.

---
## 6. Wrapper Functions (`BACKWARD::preprocess` and `BACKARD::render`)
### `BACKWARD::preprocess`
Orchestrates the backward pass for preprocessing by:
- Calling `computeCov2DCUDA` to compute gradients w.r.t. 3D covariance and means.
- Calling `preprocessCUDA` to compute remaining gradients.
### `BACKWARD::render`
Runs `renderCUDA` to compute gradients during rendering.

---
**References:**
- Karnovets, O., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *SIGGRAPH*, 2023.
- Baydin, A. G., et al. "Automatic Differentiation in Machine Learning: a Survey." *Journal of Machine Learning Research*, vol. 18, no. 153, 2018, pp. 1–43.