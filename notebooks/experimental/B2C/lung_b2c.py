import scanpy as sc
import os
import bin2cell as b2c
import celltypist
from celltypist import models
import numpy as np
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.pyplot as plt

import tifffile as tf
import cv2


def mpp_to_scalef(adata, mpp):
    """
    Compute a scale factor for a specified mpp value.

    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    mpp : ``float``
        Microns per pixel to report scale factor for.
    """
    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns["spatial"].keys())[0]
    # get original image mpp value
    mpp_source = adata.uns["spatial"][library]["scalefactors"]["microns_per_pixel"]
    # our scale factor is the original mpp divided by the new mpp
    return mpp_source / mpp


def get_mpp_coords(adata, basis="spatial", spatial_key="spatial", mpp=None):
    """
    Get an mpp-adjusted representation of spatial or array coordinates of the
    provided object. Origin in top left, dimensions correspond to ``np.array()``
    representation of image (``[:,0]`` is up-down, ``[:,1]`` is left-right).

    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to get ``"spatial"`` or ``"array"`` coordinates. The former is
        the source H&E image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``.
        Rounded coordinates will be used to represent each bin when retrieving
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value. Mandatory for GEX (``basis="array"``), if not provided
        with H&E (``basis="spatial"``) will assume full scale image.
    """
    # if we're using array coordinates, is there an mpp provided?
    if basis == "array" and mpp is None:
        raise ValueError("Need to specify mpp if working with array coordinates.")
    if basis == "spatial":
        if mpp is not None:
            # get necessary scale factor
            scalef = mpp_to_scalef(adata, mpp=mpp)
        else:
            # no mpp implies full blown H&E image, so scalef is 1
            scalef = 1
        # get the matching coordinates, rounding to integers makes this agree
        # need to reverse them here to make the coordinates match the image, as per note at start
        # multiply by the scale factor to account for possible custom mpp H&E image
        coords = (adata.obsm[spatial_key] * scalef).astype(int)[:, ::-1]
    elif basis == "array":
        # generate the pixels in the GEX image at the specified mpp
        # which actually correspond to the locations of the bins
        # easy to define scale factor as starting array mpp is 2
        scalef = 2 / mpp
        coords = np.round(adata.obs[["array_row", "array_col"]].values * scalef).astype(
            int
        )
        # need to flip axes maybe
        # need to scale up maximum appropriately
        if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
            coords[:, 0] = (
                int(adata.uns["bin2cell"]["array_check"]["row"]["max"] * scalef)
                - coords[:, 0]
            )
        if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
            coords[:, 1] = (
                int(adata.uns["bin2cell"]["array_check"]["col"]["max"] * scalef)
                - coords[:, 1]
            )
    return coords


def normalize(img):
    """
    Extremely naive reimplementation of default ``cbsdeep.utils.normalize()``
    percentile normalisation, with a lower RAM footprint than the original.

    Input
    -----
    img : ``numpy.array``
        Numpy array image to normalise
    """
    eps = 1e-20
    mi = np.percentile(img, 3)
    ma = np.percentile(img, 99.8)
    return (img - mi) / (ma - mi + eps)


def get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=0):
    """
    Get a PIL-formatted crop tuple from a provided object and coordinate
    representation.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to use ``"spatial"`` or ``"array"`` coordinates. The former is
        the source H&E image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``.
        Rounded coordinates will be used to represent each bin when retrieving
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The micron per pixel value to use. Mandatory for GEX (``basis="array"``),
        if not provided with H&E (``basis="spatial"``) will assume full scale
        image.
    buffer : ``int``, optional (default: 0)
        How many extra pixels to include to each side the cropped grid for
        extra visualisation.
    """
    # get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    # PIL crop is defined as a tuple of (left, upper, right, lower) coordinates
    # coords[:,0] is up-down, coords[:,1] is left-right
    # don't forget to add/remove buffer, and to not go past 0
    return (
        np.max([np.min(coords[:, 1]) - buffer, 0]),
        np.max([np.min(coords[:, 0]) - buffer, 0]),
        np.max(coords[:, 1]) + buffer,
        np.max(coords[:, 0]) + buffer,
    )


def scaled_if_image(
    adata,
    channel,
    mpp=1,
    crop=True,
    buffer=150,
    spatial_cropped_key="spatial_cropped",
    save_path=None,
):
    """
    Create a custom microns per pixel render of the full scale IF image for
    visualisation and downstream application. Store resulting image and its
    corresponding size factor in the object. If cropping to just the spatial
    grid, also store the cropped spatial coordinates. Optionally save to file.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution IF image provided via
        ``source_image_path`` to ``b2c.read_visium_hd()``.
    channel : ``int``
        The channel of the IF image holding the DAPI capture.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired IF image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area,
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str``, optional (default: ``"spatial_cropped"``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted
        spatial coordinates in.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for
        StarDist use).
    """
    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns["spatial"].keys())[0]
    # pull out specified channel from IF tiff via tifffile
    # pretype to float32 for space while working with plots (float16 does not)
    img = tf.imread(
        adata.uns["spatial"][library]["metadata"]["source_image_path"], key=channel
    ).astype(np.float32)
    # this can be dark, apply stardist normalisation to fix
    img = normalize(img)
    # actually cap the values - currently there are sub 0 and above 1 entries
    img[img < 0] = 0
    img[img > 1] = 1
    # crop image if necessary
    if crop:
        crop_coords = get_crop(
            adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer
        )
        # this is already capped at a minimum of 0, so can just subset freely
        # left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1] : crop_coords[3], crop_coords[0] : crop_coords[2]]
        # need to move spatial so it starts at the new crop top left point
        # spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:, 0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:, 1] -= crop_coords[1]
    # reshape image to desired microns per pixel
    # get necessary scale factor for the custom mpp
    # multiply dimensions by this to get the shrunken image size
    # multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    # need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2]) * scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # we have everything we need. store in object
    adata.uns["spatial"][library]["images"][str(mpp) + "_mpp"] = img
    # the scale factor needs to be prefaced with "tissue_"
    adata.uns["spatial"][library]["scalefactors"][
        "tissue_" + str(mpp) + "_mpp_scalef"
    ] = scalef
    if save_path is not None:
        # cv2 expects BGR channel order, we have a greyscale image
        # oh also we should make it a uint8 as otherwise stuff won't work
        cv2.imwrite(
            save_path, cv2.cvtColor((255 * img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        )


def convert_bins_to_cells(
    path_02_micron: str = "/mnt/work/RO_src/data/raw/VisiumHD/square_002um/",
    source_image_path: str = "/mnt/work/RO_src/data/raw/VisiumHD/Visium_HD_Human_Lung_Cancer_tissue_image.tif",
    save_path: str = ".",
    min_cells: int = 3,
    min_counts: int = 1,
    stardist_model: str = "2D_versatile_fluo",
    stardist_nms_thresh: float = 0.5,
    IF_CHANNEL: int = 2,
    mpp: float = 0.35,
    stardist_prob_thresh: float = 0.1,
) -> sc.AnnData:

    adata = b2c.read_visium(path=path_02_micron, source_image_path=source_image_path)
    adata.var_names_make_unique()
    adata.var_names = adata.var_names.str.upper()
    adata.var.index = adata.var.index.str.upper()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=min_counts)

    # adata.raw = adata.copy()
    # adata_raw = adata.copy()

    scaled_if_image(
        adata=adata, channel=IF_CHANNEL, mpp=mpp, save_path=f"{save_path}/if.tiff"
    )

    b2c.stardist(
        image_path=f"{save_path}/if.tiff",
        labels_npz_path=f"{save_path}/if.npz",
        stardist_model=stardist_model,
        prob_thresh=stardist_prob_thresh,
    )
    b2c.insert_labels(
        adata=adata,
        labels_npz_path=f"{save_path}/if.npz",
        basis="spatial",
        spatial_key="spatial_cropped",
        mpp=mpp,
        labels_key="labels_if",
    )

    b2c.destripe(adata=adata)

    b2c.expand_labels(
        adata=adata, labels_key="labels_if", expanded_labels_key="labels_if_expanded"
    )
    b2c.grid_image(
        adata=adata,
        val="n_counts_adjusted",
        mpp=mpp,
        sigma=5,
        save_path=f"{save_path}/if_gex.tiff",
    )
    b2c.stardist(
        image_path=f"{save_path}/if_gex.tiff",
        labels_npz_path=f"{save_path}/if_gex.npz",
        stardist_model=stardist_model,
        prob_thresh=stardist_prob_thresh,
        nms_thresh=stardist_nms_thresh,
    )
    b2c.insert_labels(
        adata=adata,
        labels_npz_path="./if_gex.npz",
        basis="array",
        mpp=mpp,
        labels_key="labels_if_gex",
    )

    b2c.salvage_secondary_labels(
        adata=adata,
        primary_label="labels_if_expanded",
        secondary_label="labels_if_gex",
        labels_key="labels_if_joint",
    )

    # sc.pl.spatial(bdata, color=[None, "labels_joint_source", "labels_joint"], img_key="0.5_mpp", basis="spatial_cropped")
    adata_b2c: sc.AnnData = b2c.bin_to_cell(
        adata=adata,
        labels_key="labels_if_joint",
        spatial_keys=["spatial", "spatial_cropped"],
    )

    adata_b2c.write_h5ad(filename=f"{save_path}/adata_b2c.h5ad")
    adata.write_h5ad(f"{save_path}/adata.h5ad")

    return adata_b2c
    # cell_mask = (
    #     (adata_b2c.obs["array_row"] >= 1450)
    #     & (adata_b2c.obs["array_row"] <= 1550)
    #     & (adata_b2c.obs["array_col"] >= 250)
    #     & (adata_b2c.obs["array_col"] <= 450)
    # )

    # ddata = adata_b2c[cell_mask]

    # sc.pl.spatial(
    #     adata=ddata, color=["bin_count", "labels_if_joint_source"], basis="spatial_cropped"
    # )


if __name__ == "__main__":
    adata_b2c: sc.AnnData = convert_bins_to_cells()
