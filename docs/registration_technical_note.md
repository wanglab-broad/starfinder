# Landmark-Based Registration for Sparse Fluorescence Spot Images

**Addressing Limitations of Intensity-Based Diffeomorphic Registration**

*February 2026*

---

## 1. Problem Statement

Spatial transcriptomics and multiplexed FISH techniques produce images characterized by **sparse, punctate fluorescence signals** (spots) against a predominantly dark background. Accurate image registration is essential for aligning data across imaging rounds, channels, or modalities. However, conventional intensity-based registration methods perform poorly on this class of images due to the sparsity of informative signal.

This document summarizes our findings from systematic benchmarking of diffeomorphic registration on synthetic spot images, identifies the root causes of poor performance, and proposes a landmark-based registration pipeline using **Coherent Point Drift (CPD)** as a robust alternative.

## 2. Benchmark Setup

### 2.1 Synthetic Deformation Model

To evaluate registration methods under controlled conditions, we generated synthetic deformation fields with known ground truth. Three deformation types were implemented:

- **Polynomial:** Smooth, globally varying warps using low-order polynomial basis functions (constant + linear + bilinear cross-terms, 6 coefficients per axis). This represents tissue-scale deformation such as slow drift or thermal expansion.
- **Gaussian:** A single localized Gaussian bump with random center and direction, simulating a focal distortion such as a bubble or local tissue deformation.
- **Multi-point:** Multiple independent Gaussian bumps, representing complex local deformations arising from multiple sources.

Each deformation field is normalized to a specified maximum displacement and applied to the moving image to create a ground-truth warped pair.

### 2.2 Registration Method Tested

We evaluated **diffeomorphic registration** with 50 iterations and a smoothing parameter σ = 1.0 (denoted `diffeomorphic_iter50_s1.0`). The test pair was `polynomial_small`, representing a low-magnitude polynomial deformation.

## 3. Results and Diagnosis

### 3.1 Registration Quality Metrics

| Metric | Before | After |
|--------|--------|-------|
| MAD (bright pixels) | 0.248 | 0.195 (↓22%) |
| NCC | 0.029 | 0.103 |
| SSIM | 0.382 | 0.463 |
| IoU | 0.030 | 0.118 |
| Runtime | — | 27.57 s |

*Table 1. Registration metrics before and after diffeomorphic registration on the polynomial_small test case. While all metrics improved, absolute values remain far below acceptable alignment quality (e.g., NCC ≈ 0.1, IoU ≈ 0.12).*

### 3.2 Visual Inspection

![Registration inspection panel](inspection_diffeomorphic_iter50_s1_0_polynomial_small.png)

*Figure 1. Registration inspection panel. Left two panels show overlay of reference (green) and moving/registered (magenta) images before and after registration. Right two panels show absolute difference maps. Persistent green/magenta separation in the "After" panel indicates substantial residual misalignment.*

### 3.3 Root Cause Analysis

The polynomial deformation used in this test is notably simple: a low-order smooth warp with only 6 coefficients per axis (constant, linear, and bilinear cross-terms). This class of deformation should be easily captured by diffeomorphic or even affine registration methods. The poor result is therefore **not attributable to deformation complexity**, but rather to the **nature of the image data itself**.

The fundamental issue is that intensity-based registration relies on smooth image gradients to drive the optimization. Sparse spot images present three specific challenges:

1. **Gradient sparsity:** Most of the image is dark background with near-zero gradient. The optimizer has no signal to guide alignment in the vast regions between spots.
2. **Discontinuous cost landscape:** The cost function has narrow, isolated wells around each spot, with flat plateaus in between. This creates many local minima and makes gradient-based optimization unreliable.
3. **Low starting similarity:** The very low initial NCC (0.029) means the similarity metric can barely distinguish aligned from misaligned states, giving the optimizer a weak and noisy signal from the start.

Pre-smoothing the images (to create broader intensity basins) and multi-resolution pyramids were also tested but **did not yield efficient improvement** for this class of data, motivating a fundamentally different approach.

## 4. Proposed Solution: Landmark-Based Registration

Rather than operating on pixel intensities, landmark-based registration extracts the informative content (spot positions) and works directly with point clouds. This approach inherently avoids the gradient sparsity problem.

### 4.1 Pipeline Overview

1. **Spot detection:** Extract spot centroids from both reference and moving images, producing two point clouds. Standard blob detection methods (LoG, DoG) or existing decoding pipelines can be used.
2. **Point cloud matching:** Establish correspondences between the two point clouds using Coherent Point Drift (CPD). CPD is preferred because it handles partial overlap, requires no initial correspondences, and is robust to outliers.
3. **Transform fitting:** Fit a smooth spatial transform from matched pairs using RBF interpolation (thin-plate spline kernel) or polynomial fitting.
4. **Warp application:** Apply the fitted transform to warp the full moving image or transform spot coordinates directly.

### 4.2 Coherent Point Drift (CPD)

CPD is a probabilistic point set registration algorithm that treats one point set as **Gaussian Mixture Model (GMM) centroids** and the other as **observed data points**. It iteratively moves the centroids to maximize the likelihood of the data through an Expectation-Maximization (EM) procedure.

#### 4.2.1 Algorithm

Given reference points Y (N points) and moving points X (M points), CPD iterates:

- **E-step:** Compute soft (probabilistic) assignments between each data point in X and each GMM centroid in Y. Unlike hard nearest-neighbor matching, each moving point has a weighted association with multiple reference points.
- **M-step:** Update centroid positions (warp Y) to maximize data likelihood. The transform can be rigid, affine, or non-rigid depending on the variant used.

#### 4.2.2 Handling Partial Overlap

The critical feature for sparse spot data is CPD's outlier model. The GMM includes a uniform distribution component, controlled by a weight parameter w ∈ [0, 1]:

$$P(x) = w \cdot \frac{1}{V} + (1 - w) \cdot \sum_n \text{Gaussian}_n(x)$$

This means:

- Points in X with no counterpart in Y are explained by the uniform component and treated as outliers, without corrupting the estimated transform.
- Points in Y with no counterpart in X simply receive low posterior responsibility and remain approximately stationary.
- Setting `w = 0.3` indicates an expectation of ~30% unmatched points. This should be tuned based on the expected overlap between imaging rounds.

#### 4.2.3 Non-Rigid CPD

For spatially varying deformations (polynomial, Gaussian bump, multi-point), the non-rigid variant of CPD parameterizes the displacement as a linear combination of Gaussian kernels centered at the GMM centroids:

$$\text{displacement}(y_n) = \sum_m G_{nm} \cdot w_m$$

where **G** is a Gaussian kernel matrix enforcing spatial coherence (the "coherent" in CPD): nearby points are constrained to move similarly. The kernel bandwidth (β) controls the smoothness of the deformation field.

#### 4.2.4 Key Parameters

| Parameter | Role | Guidance |
|-----------|------|----------|
| `w` | Outlier weight | Set to 1 − (expected overlap fraction). Critical for partial overlap scenarios. |
| `β` | Kernel width | Controls spatial coherence scale. Large β for smooth global warps (polynomial); smaller for localized deformations (Gaussian bumps). |
| `α` | Regularization | Penalizes deformation magnitude. Higher values bias toward rigid-like behavior. |

*Table 2. Key CPD parameters and tuning guidance.*

#### 4.2.5 Example Usage

```python
from pycpd import DeformableRegistration
import numpy as np

# spots_ref: (N, 3) array of reference spot coordinates
# spots_mov: (M, 3) array of moving spot coordinates

reg = DeformableRegistration(
    X=spots_ref,       # target (data points)
    Y=spots_mov,       # source (GMM centroids, will be moved)
    w=0.3,             # outlier weight — tune based on expected overlap
    alpha=2.0,          # regularization (higher = smoother)
    beta=2.0,           # Gaussian kernel width (higher = more coherent)
    tolerance=1e-5,
    max_iterations=150,
)

transformed_points, params = reg.register()
```

### 4.3 Generating a Dense Displacement Field

After CPD produces matched point pairs, a dense displacement field is needed to warp the full image. Two approaches are recommended: (1) evaluate the learned CPD kernel weights on a regular grid, or (2) use `scipy.interpolate.RBFInterpolator` with a thin-plate spline kernel to interpolate the displacement from matched landmarks to all pixels. The second approach is simpler and provides more control over smoothness via the regularization parameter.

```python
from scipy.interpolate import RBFInterpolator

# matched_src, matched_dst: (N, 3) arrays of corresponding points
displacements = matched_dst - matched_src

# Fit one interpolator per axis
interpolators = []
for axis in range(3):
    rbf = RBFInterpolator(
        matched_src,
        displacements[:, axis],
        kernel="thin_plate_spline",
        smoothing=1.0,  # regularization
    )
    interpolators.append(rbf)
```

### 4.4 Computational Considerations

Standard CPD has O(NM) complexity per iteration due to the full correspondence matrix. For point clouds up to a few thousand spots, this is manageable. For larger datasets (tens of thousands of spots), consider: (1) subsampling via farthest-point sampling, (2) fast CPD implementations using the Nyström approximation or Fast Gauss Transform (available in the `probreg` Python package), or (3) GPU-accelerated variants.

## 5. Alternative Matching Approaches

While CPD is the recommended primary approach, several alternatives may be useful in specific scenarios:

- **RANSAC + nearest-neighbor:** Match spots to nearest neighbors, then iteratively reject outliers. Effective when displacements are smaller than the mean inter-spot distance. Available via `skimage.measure.ransac`.
- **Iterative Closest Point (ICP):** Classic approach; start with rigid ICP for coarse alignment, then refine with non-rigid. Available in `open3d` and `probreg`.
- **Optimal transport:** Globally optimal 1-to-1 matching via `scipy.optimize.linear_sum_assignment` on the pairwise distance matrix. Produces clean correspondences but is expensive for large point sets and does not natively handle partial overlap.

## 6. Recommendations and Next Steps

1. **Implement CPD + RBF pipeline:** Use non-rigid CPD for point cloud matching and RBFInterpolator for dense field generation as the primary registration method for sparse spot images.
2. **Benchmark against ground truth fields:** Since the synthetic benchmark includes known deformation fields, evaluate the recovered displacement field directly against ground truth, rather than relying solely on image-level metrics (NCC, SSIM) which are inherently noisy for sparse data.
3. **Tune outlier weight per experiment:** Estimate the expected spot overlap between rounds and set the CPD outlier weight accordingly. This is the most impactful parameter for real-world performance.
4. **Test across all deformation types:** Validate on polynomial, Gaussian, and multi-point deformations. Adjust the kernel bandwidth (β) parameter based on the expected spatial scale of deformation.
5. **Consider hybrid approaches:** For maximum robustness, use landmark-based CPD for initial coarse alignment, optionally followed by a constrained intensity-based refinement on smoothed images.

## References

1. Myronenko, A. & Song, X. (2010). Point Set Registration: Coherent Point Drift. *IEEE TPAMI*, 32(12), 2262–2275.
2. pycpd: Python implementation of CPD. https://github.com/siavashk/pycpd
3. probreg: Probabilistic point cloud registration. https://github.com/neka-nat/probreg
