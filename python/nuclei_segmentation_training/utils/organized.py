import numpy as np
import skfmm


from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes
from skimage.feature import peak_local_max

from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import label


def bwdistgeodesic(
    seeds, mask, bakround_value=np.Inf, seed2zero=True, diminsion_weights=None
):
    """Compute a geodesic distance transform constrained to a binary mask.

    This uses the Fast Marching Method implementation in `skfmm.distance` to compute
    distance from a set of seed points, but *only allowing propagation inside* the
    foreground region of `mask` (i.e., you "walk" only where `mask` is True).
    Outside the mask, the returned value is set to `bakround_value`.

    Notes
    -----
    - The function name/arguments keep the original spelling (e.g. `bakround_value`,
      `diminsion_weights`) to avoid breaking existing calls.
    - Internally we create a signed level-set field `m` where seeds are negative
      and non-seed positions are positive, then mask out the background so FMM
      only solves inside the foreground.

    Parameters
    ----------
    seeds : ndarray (bool or 0/1)
        Seed locations (True/1 at seeds).
    mask : ndarray (bool or 0/1)
        Foreground region where the distance is allowed to propagate.
    bakround_value : float, optional
        Value assigned to positions where `mask` is False. Default: +inf.
    seed2zero : bool, optional
        If True, force distances at seed locations to 0 (skfmm returns negative
        values inside the negative region). Default: True.
    diminsion_weights : sequence of float, optional
        Per-axis weights for anisotropic voxel spacing. Passed as `dx=` to skfmm as
        `1/weights` to match the original code's convention.

    Returns
    -------
    distance : ndarray
        Geodesic distance image (finite inside mask, `bakround_value` outside).
    """

    # Build a signed level-set: seeds are "inside" (negative), elsewhere positive.
    # skfmm.distance computes the distance to the zero level-set of this field.
    m = np.ones_like(seeds, dtype=np.float32)
    m[seeds > 0] = -1

    # Prevent propagation outside `mask` by masking those positions out.
    m = np.ma.masked_array(m, mask == 0)

    # Compute distance; dx encodes voxel spacing (anisotropy).
    if diminsion_weights:
        distance = skfmm.distance(m, dx=1 / np.array(diminsion_weights))
    else:
        distance = skfmm.distance(m)

    # Convert masked array result back to plain ndarray.
    distance = distance.data

    # skfmm returns negative values inside the negative region (the seeds).
    # For segmentation we usually want seeds to have distance 0.
    if seed2zero:
        distance[distance < 0] = 0

    # Assign a constant value outside the allowed region.
    distance[mask == 0] = bakround_value

    return distance


def bwdist(mask, diminsion_weights=None, out2zero=True):
    """Compute a distance transform (via Fast Marching) from a binary mask.

    This is similar in spirit to a Euclidean distance transform, but computed using
    `skfmm.distance` on a signed field derived from `mask`.

    Parameters
    ----------
    mask : ndarray (bool or 0/1)
        Foreground region (True/1) for which distances are computed.
    diminsion_weights : sequence of float, optional
        Per-axis weights/spacing control (anisotropy).
    out2zero : bool, optional
        If True, clip negative distances (outside region) to 0. Default: True.

    Returns
    -------
    distance : ndarray
        Distance map. If `out2zero` is True, values outside the mask become 0.
    """

    # Signed field: inside mask is positive, outside is negative.
    m = -1 * np.ones_like(mask, dtype=np.float32)
    m[mask > 0] = 1

    # Run fast marching distance with optional anisotropic spacing.
    if diminsion_weights:
        distance = skfmm.distance(m, dx=1 / np.array(diminsion_weights))
    else:
        distance = skfmm.distance(m)

    # Optionally zero-out values outside the object (negative distances).
    if out2zero:
        distance[distance < 0] = 0

    return distance


def balloon(mask, strel):
    """Split/regularize connected objects using a geodesic watershed ("balloon").

    High-level idea:
    - Dilate the mask to create a slightly larger connected region.
    - Compute *geodesic distance* from the original mask inside the dilated region.
    - Use watershed seeded by the original connected components to expand them
      through the dilated region (like inflating balloons).

    Parameters
    ----------
    mask : ndarray (bool)
        Binary mask containing objects (usually after an initial split).
    strel : ndarray (bool)
        Structuring element used for dilation (e.g., an ellipsoid).

    Returns
    -------
    labels : ndarray (int)
        Labeled image after the balloon expansion/watershed.
    """

    # Create a slightly expanded region where labels are allowed to grow.
    mask_conected = binary_dilation(mask, strel)

    # Distance from the original object, but constrained within the dilated region.
    # The [1, 1, 3] weights reflect anisotropic z-spacing in the original workflow.
    D = bwdistgeodesic(mask, mask_conected, diminsion_weights=[1, 1, 3])

    # Use connected components of the original mask as watershed seeds.
    labeled_seeds = label(mask)[0]

    # Watershed expands each seed region inside the connected mask.
    labels = watershed(D, labeled_seeds, mask=mask_conected, watershed_line=True)

    return labels


def split_nuclei(mask, minimal_nuclei_size, h, sphere, min_dist):
    """Split merged nuclei in a 3D binary mask using distance + watershed seeding.

    Pipeline overview:
    1) Clean the binary mask (remove tiny components and fill small holes).
    2) Compute a distance map inside the mask.
    3) Find likely nucleus centers using two complementary criteria:
       - `h_maxima(D, h)` suppresses weak local maxima (noise).
       - `peak_local_max(D, min_distance=...)` enforces spatial separation.
       Seeds are the intersection of both.
    4) Apply watershed on `-D` to split merged objects (basin flooding).
    5) Post-process seeds and run a geodesic watershed to refine boundaries.

    Parameters
    ----------
    mask : ndarray (bool)
        Binary nuclei mask.
    minimal_nuclei_size : int
        Minimum object size kept during cleanup.
    h : float
        h-parameter for `h_maxima` (higher => fewer maxima).
    sphere : ndarray (bool)
        Structuring element (kept for compatibility; not used in current code path).
    min_dist : int
        Minimum distance between local maxima seeds (in pixels).

    Returns
    -------
    labelss : ndarray (int)
        Labeled nuclei after splitting/refinement.
    """

    # --- Mask cleanup ---
    # Remove tiny objects and fill small holes to stabilize the distance transform.
    mask = remove_small_objects(mask, minimal_nuclei_size)
    mask = remove_small_holes(mask, minimal_nuclei_size)

    # --- Distance transform ---
    # Compute distance inside the mask. If voxels are anisotropic, you would pass
    # different weights, but here it's isotropic (1,1,1).
    D = bwdist(mask, diminsion_weights=(1, 1, 1))

    # --- Seed detection ---
    # 1) Suppress shallow maxima (noise).
    maxima2 = h_maxima(D, h)

    # 2) Enforce that peaks are at least `min_dist` apart.
    peak_idx = peak_local_max(D, min_distance=min_dist, exclude_border=False)
    maxima1 = np.zeros_like(D, dtype=bool)
    maxima1[tuple(peak_idx.T)] = True

    # Combine both criteria for more robust seeds.
    labeled_maxima, num = label(maxima1 & maxima2)

    # --- First watershed split ---
    # Watershed on -D splits objects at narrow necks between distance peaks.
    labels = watershed(-D, labeled_maxima, mask=mask, watershed_line=True)

    # Keep only sufficiently large seed regions (removes spurious splits).
    seeds = remove_small_objects(labels > 0, 30000)

    # Slight erosion to separate touching seeds before geodesic refinement.
    seeds = binary_erosion(seeds)

    # --- Geodesic refinement ---
    # Compute geodesic distances from seeds constrained to the original mask.
    DD = bwdistgeodesic(seeds, mask, diminsion_weights=[1, 1, 1])

    # Label each seed component to use as distinct watershed markers.
    labeled_seeds, num = label(seeds)

    # Final watershed refines the split boundaries in a mask-constrained way.
    labelss = watershed(DD, labeled_seeds, mask=mask, watershed_line=True)

    return labelss
