"""Nested hybrid grid creation.

Create a merged grid where one region is refined (subdivided) and stitched
back into the original grid via Non-Neighbour Connections (NNCs).

Public API
----------
create_nested_hybrid_grid : function
    Build a nested hybrid grid from a coarse grid, a region property,
    and a refinement specification.
nnc_to_gridproperty : function
    Convert NNC transmissibility DataFrames to GridProperty instances.
nnc_to_flowsimulator_input : function
    Write NNC transmissibilities to a flow-simulator input file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xtgeo


if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _crop_for_region(
    grid: xtgeo.Grid,
    region: xtgeo.GridProperty,
    region_id: int,
) -> tuple[xtgeo.Grid, tuple[int, int, int]]:
    """Crop *grid* to the bounding box of *region_id*.

    Returns (cropped_grid, crop_origin) where *crop_origin* is the 0-based
    (i, j, k) offset of the crop start inside the original grid.
    """
    region_values = region.values
    region_indices = np.where(region_values == region_id)

    imin = int(region_indices[0].min() + 1)
    imax = int(region_indices[0].max() + 1)
    jmin = int(region_indices[1].min() + 1)
    jmax = int(region_indices[1].max() + 1)
    kmin = int(region_indices[2].min() + 1)
    kmax = int(region_indices[2].max() + 1)

    _logger.info(
        "Region %d bounding box (1-based): i=%d-%d, j=%d-%d, k=%d-%d",
        region_id,
        imin,
        imax,
        jmin,
        jmax,
        kmin,
        kmax,
    )

    cropped_grid = grid.copy()
    cropped_grid.crop((imin, imax), (jmin, jmax), (kmin, kmax), props="all")
    _logger.info("Cropped grid dimensions: %s", cropped_grid.dimensions)

    return cropped_grid, (imin - 1, jmin - 1, kmin - 1)


def _find_boundary_faces(
    region_prop: xtgeo.GridProperty,
    target_region: int,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int], str]]:
    """Find cell faces on the boundary between *target_region* and other active regions.

    Returns a list of ``(outside_ijk, inside_ijk, face_dir)`` where indices
    are 0-based and *face_dir* is one of ``'i+', 'i-', 'j+', 'j-', 'k+', 'k-'``.
    """
    filled = np.ma.filled(region_prop.values, fill_value=-1).astype(int)
    ni, nj, nk = filled.shape

    in_target = filled == target_region
    outside_active = (filled != target_region) & (filled != -1)

    faces: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = []

    # i+
    mask = in_target[: ni - 1, :, :] & outside_active[1:ni, :, :]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i + 1), int(j), int(k)), (int(i), int(j), int(k)), "i+"))

    # i-
    mask = in_target[1:ni, :, :] & outside_active[: ni - 1, :, :]
    for idx, j, k in np.argwhere(mask):
        faces.append(((int(idx), int(j), int(k)), (int(idx + 1), int(j), int(k)), "i-"))

    # j+
    mask = in_target[:, : nj - 1, :] & outside_active[:, 1:nj, :]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i), int(j + 1), int(k)), (int(i), int(j), int(k)), "j+"))

    # j-
    mask = in_target[:, 1:nj, :] & outside_active[:, : nj - 1, :]
    for i, jdx, k in np.argwhere(mask):
        faces.append(((int(i), int(jdx), int(k)), (int(i), int(jdx + 1), int(k)), "j-"))

    # k+
    mask = in_target[:, :, : nk - 1] & outside_active[:, :, 1:nk]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i), int(j), int(k + 1)), (int(i), int(j), int(k)), "k+"))

    # k-
    mask = in_target[:, :, 1:nk] & outside_active[:, :, : nk - 1]
    for i, j, kdx in np.argwhere(mask):
        faces.append(((int(i), int(j), int(kdx)), (int(i), int(j), int(kdx + 1)), "k-"))

    _logger.info("Found %d boundary faces for region %d", len(faces), target_region)
    return faces


def _compute_nnc_table(
    region_prop: xtgeo.GridProperty,
    target_region_id: int,
    crop_origin: tuple[int, int, int],
    refinement: tuple[int, int, int],
    coarse_ncol: int,
) -> pd.DataFrame:
    """Compute NNC cell-pair mapping between mother and refined cells.

    For each boundary face between the target region and the surrounding mother
    cells, this function determines which refined sub-cells in the merged grid
    connect to which mother cell, and through which face direction.

    The mapping is purely topological (index-based) — no geometric computation
    is performed here.  The resulting table is intended to be passed to
    :meth:`xtgeo.Grid.get_transmissibilities` so that it can compute the
    actual transmissibility for each cell pair.

    Convention:
        - ``I1, J1, K1`` is always the **mother** cell (1-based, merged grid).
        - ``I2, J2, K2`` is always the **refined** cell (1-based, merged grid).
        - ``DIRECTION`` is from the mother cell's perspective (e.g. ``"I+"``
          means looking in the positive I-direction from the mother cell
          you reach the refined cell).

    Args:
        region_prop: Region property on the original (unmodified) grid.
        target_region_id: Region value that was refined.
        crop_origin: 0-based ``(i0, j0, k0)`` origin of the crop box.
        refinement: ``(rcol, rrow, rlay)`` refinement factors.
        coarse_ncol: Number of columns in the coarse grid (grid1 in the merge).

    Returns:
        A DataFrame with columns ``I1, J1, K1, I2, J2, K2, DIRECTION``.
    """
    faces = _find_boundary_faces(region_prop, target_region_id)

    rcol, rrow, rlay = refinement
    i0, j0, k0 = crop_origin
    # In the merged grid, grid2 (refined) starts after a 1-column gap:
    i_offset = coarse_ncol + 1

    rows: list[dict[str, int | str]] = []

    for outside_ijk, inside_ijk, face_dir in faces:
        # outside_ijk = mother cell (0-based in original/merged grid)
        mi, mj, mk = outside_ijk

        # inside_ijk = target cell (0-based in original grid) → cropped coords
        ci = inside_ijk[0] - i0
        cj = inside_ijk[1] - j0
        ck = inside_ijk[2] - k0

        # Determine direction from mother and which refined cells lie on the face.
        # face_dir is from the *inside* (target) cell's perspective;
        # the mother's perspective is the opposite sign.
        #
        # For I-faces: the varying refined indices are J and K  (rrow × rlay cells)
        # For J-faces: the varying refined indices are I and K  (rcol × rlay cells)
        # For K-faces: the varying refined indices are I and J  (rcol × rrow cells)
        ref_is: Iterable[int]
        ref_js: Iterable[int]
        ref_ks: Iterable[int]

        if face_dir == "i-":
            # Target at higher I than mother → mother's I+ face
            direction = "I+"
            ref_is = [ci * rcol]  # first i-column of refined block (I- face)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "i+":
            # Target at lower I than mother → mother's I- face
            direction = "I-"
            ref_is = [ci * rcol + rcol - 1]  # last i-column (I+ face)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "j-":
            # Target at higher J than mother → mother's J+ face
            direction = "J+"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = [cj * rrow]  # first j-row (J- face)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "j+":
            # Target at lower J than mother → mother's J- face
            direction = "J-"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = [cj * rrow + rrow - 1]  # last j-row (J+ face)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "k-":
            # Target at higher K than mother → mother's K+ face
            direction = "K+"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = [ck * rlay]  # first k-layer (K- face)
        elif face_dir == "k+":
            # Target at lower K than mother → mother's K- face
            direction = "K-"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = [ck * rlay + rlay - 1]  # last k-layer (K+ face)
        else:
            raise ValueError(f"Unexpected face direction: {face_dir!r}")

        for ri in ref_is:
            for rj in ref_js:
                for rk in ref_ks:
                    rows.append(
                        {
                            "I1": mi + 1,
                            "J1": mj + 1,
                            "K1": mk + 1,
                            "I2": ri + i_offset + 1,
                            "J2": rj + 1,
                            "K2": rk + 1,
                            "DIRECTION": direction,
                        }
                    )

    _logger.info(
        "NNC table: %d cell pairs from %d boundary faces", len(rows), len(faces)
    )
    return pd.DataFrame(rows, columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])


def _set_actnum_by_region(
    grid: xtgeo.Grid,
    region_prop: xtgeo.GridProperty,
    target_region: int,
    *,
    invert: bool = False,
) -> None:
    """Deactivate cells based on region membership."""
    actnum = grid.get_actnum()
    region_values = region_prop.values

    mask = region_values != target_region if invert else region_values == target_region

    _logger.info(
        "Deactivating %d cells (region %s %d)",
        np.sum(mask),
        "!=" if invert else "==",
        target_region,
    )
    actnum.values[mask] = 0
    grid.set_actnum(actnum)

def _modify_upscaling_mapping(
    upscale: tuple[ xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty ],
    region: xtgeo.GridProperty,
    target_region_id: int,
    refinement: tuple[int, int, int],
    offset: tuple[int, int, int],
    grid2_dims: tuple[int, int, int],
) -> tuple[ xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty]:
    """Update a cell mapping for upscaling
        upscale: (
            I_property - xtgeo.GridProperty mapping geogrid cells to merged grid I
            J_property - xtgeo.GridProperty mapping geogrid cells to merged grid J
            K_property - xtgeo.GridProperty mapping geogrid cells to merged grid K
            I refinement - xtgeo.GridProperty on input grid with number geogrid per in I
            J refinement - xtgeo.GridProperty on input grid with number geogrid per in J
            K refinement - xtgeo.GridProperty on input grid with number geogrid per in K
            )
        region: input region (before merging)
        target_region_id: region to be refined
        refinement: refinement (i,j,k) to be applied per cell in target region
        offset: start cell of the refinement region
        grid2_dims: (columns, rows, layers) in refined grid
    """
    
    # shift region from input grid to geogrid to know which cells are refined or not
    imap, jmap, kmap, iref, jref, kref = upscale

    iv=imap.values.reshape(-1)-1
    jv=jmap.values.reshape(-1)-1
    kv=kmap.values.reshape(-1)-1
    irv=iref.values
    jrv=jref.values
    krv=kref.values
    
    ivo=iv.copy()
    jvo=jv.copy()
    kvo=kv.copy()
 
    cn=iv*(jv.max()+1)*(kv.max()+1)+jv*(kv.max()+1)+kv
    
    region2=np.ma.masked_array(np.zeros(len(iv)),mask=cn.mask)
    region2[np.where(cn.mask==False)]=region.values.reshape(-1)[cn[np.where(cn.mask==False)].astype(np.int32)]
    
    oi, oj, ok = offset
    ri, rj, rk = refinement
    di, dj, dk = grid2_dims
  
    # create a mapping from old layer to new layer where there is no refinement
    lmap = np.arange(kv.max()+1,dtype=np.int32)
    lmap = lmap + np.where(
        lmap < ok,
        0,
        (rk - 1)
        * np.minimum(int(dk / rk), lmap - ok),
    )
    kvo[np.where((cn.mask == False) & (region2 != target_region_id))]=lmap[kv[np.where((cn.mask==False) & (region2 != target_region_id))].astype(np.int32)]
 

#    # create a mapping from old layer to new layer where there is a refinement
#    # split by refinement level in ijk directions to simplify mapping
    for uri in np.unique(irv[np.where((irv.mask==False) & (region.values==target_region_id))]):
        for urj in np.unique(jrv[np.where((irv.mask==False) & (region.values==target_region_id) & (irv==uri))]):
            for urk in np.unique(krv[np.where((irv.mask==False) & (region.values==target_region_id) & (irv==uri) & (jrv==urj))]):
                cijkr = np.argwhere((irv.mask==False) & (region.values==target_region_id) & (irv==uri) & (jrv==urj) & (krv==urk))
                cnr = cijkr[:,0]*(jv.max()+1)*(kv.max()+1)+cijkr[:,1]*(kv.max()+1)+cijkr[:,2]

                il2 = (np.repeat(np.arange( di/ri, dtype=np.float32 ) + oi,ri*dj*dk)).reshape((di,dj,dk))
                il2 = np.repeat( il2, uri/ri, axis = 0)
                il2 = np.repeat( il2, urj/rj, axis = 1)
                il2 = np.repeat( il2, urk/rk, axis = 2)
                il2 = il2.reshape(-1)

                jl2 = np.swapaxes((np.repeat(np.arange( dj/rj, dtype=np.float32 ) + oj,rj*di*dk)).reshape((dj,di,dk)),0,1)
                jl2 = np.repeat( jl2, uri/ri, axis = 0)
                jl2 = np.repeat( jl2, urj/rj, axis = 1)
                jl2 = np.repeat( jl2, urk/rk, axis = 2)
                jl2 = jl2.reshape(-1)

                kl2 = np.swapaxes((np.repeat(np.arange( dk/rk, dtype=np.float32 ) + ok,rk*di*dj)).reshape((dk,dj,di)),0,2)
                kl2 = np.repeat( kl2, uri/ri, axis = 0)
                kl2 = np.repeat( kl2, urj/rj, axis = 1)
                kl2 = np.repeat( kl2, urk/rk, axis = 2)
                kl2 = kl2.reshape(-1)

                cl2 = il2*(jv.max()+1)*(kv.max()+1)+jl2*(kv.max()+1)+kl2
                 
                lmap2 = np.arange( dk, dtype=np.float32 ) + ok
                lmap2 = np.tile( lmap2, di * dj ).reshape((di,dj,dk))
                lmap2 = np.repeat( lmap2, uri/ri, axis = 0) 
                lmap2 = np.repeat( lmap2, urj/rj, axis = 1)
                lmap2 = np.repeat( lmap2, urk/rk, axis = 2)

                cmap2 = np.arange( di, dtype=np.float32 ) + iv.max() + 2
                cmap2 = np.repeat( cmap2, dj * dk ).reshape((di,dj,dk))
                cmap2 = np.repeat( cmap2, uri/ri, axis = 0)
                cmap2 = np.repeat( cmap2, urj/rj, axis = 1)
                cmap2 = np.repeat( cmap2, urk/rk, axis = 2) 

                rmap2 = np.arange( dj, dtype=np.float32 ) 
                rmap2 = np.swapaxes(np.repeat(rmap2, di*dk).reshape((dj,di,dk)),0,1)
                rmap2 = np.repeat( rmap2, uri/ri, axis = 0)
                rmap2 = np.repeat( rmap2, urj/rj, axis = 1)
                rmap2 = np.repeat( rmap2, urk/rk, axis = 2)

                lmap2 = lmap2.reshape(-1)
                kvo[np.isin(cn,cnr)]=lmap2[np.isin(cl2,cnr)]

                cmap2 = cmap2.reshape(-1)
                ivo[np.isin(cn,cnr)]=cmap2[np.isin(cl2,cnr)]

                rmap2 = rmap2.reshape(-1)
                jvo[np.isin(cn,cnr)]=rmap2[np.isin(cl2,cnr)] 


    imap.values=ivo.reshape(imap.values.shape).astype(np.float32)+1.0
    jmap.values=jvo.reshape(imap.values.shape).astype(np.float32)+1.0
    kmap.values=kvo.reshape(imap.values.shape).astype(np.float32)+1.0

#region != target_region_id
#no change to imap or jmap
#kmap update layering

#region == target_region_id
# need to know refinement per cell in geogrid and simgrid
# update imap and jmap based on offset and refinement
# update kmap based on layering

    return ( imap, jmap, kmap )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_nested_hybrid_grid(
    grid: xtgeo.Grid,
    region: xtgeo.GridProperty,
    target_region_id: int,
    refinement: tuple[int, int, int],
    upscaling: tuple[ xtgeo.GridProperty,  xtgeo.GridProperty,  xtgeo.GridProperty, xtgeo.GridProperty,  xtgeo.GridProperty,  xtgeo.GridProperty ] | None = None,
) -> tuple[xtgeo.Grid, pd.DataFrame,  tuple[ xtgeo.GridProperty,  xtgeo.GridProperty,  xtgeo.GridProperty ] | None ]:
    """Create a nested hybrid grid by refining one region and merging it back.

    The cells belonging to *target_region_id* are replaced by a refined
    (subdivided) version of the same region.  A ``NEST_ID`` discrete property
    is attached to the merged grid, encoding the nested hybrid structure:

    - ``NEST_ID == 1``: coarse (mother) grid cells.
    - ``NEST_ID == 2``: refined grid cells.

    In addition, a **NNC mapping table** is returned that lists every
    mother ↔ refined cell pair that should be connected by a Non-Neighbour
    Connection (NNC).  The table is derived from the topological knowledge
    available at merge time (which original cell was refined and how its
    sub-cells map into the merged grid).

    The table columns are:

    - ``I1, J1, K1``: mother cell indices (1-based) in the merged grid.
    - ``I2, J2, K2``: refined cell indices (1-based) in the merged grid.
    - ``DIRECTION``: face direction from the mother cell's perspective
      (``I+``, ``I-``, ``J+``, ``J-``, ``K+``, ``K-``).

    This table can be passed to
    :meth:`xtgeo.Grid.get_transmissibilities` to compute NNC
    transmissibilities for the specified cell pairs.

    Args:
        grid: The original coarse grid.
        region: A :class:`xtgeo.GridProperty` whose values identify the
            regions (e.g. an integer region parameter).
        target_region_id: The region value to refine.
        refinement: ``(ncol, nrow, nlay)`` refinement factors.
        geo_grid: The geogrid with upscaling mapping attached
        upscale_props: (I,J,K) name of mapping properties to update

    Returns:
        A tuple ``(merged_grid, nnc_table, geogrid)`` where *merged_grid* is a new
        :class:`xtgeo.Grid` with the refined region stitched back into the
        coarse grid and *nnc_table* is a :class:`pandas.DataFrame` mapping
        mother cells to their connected refined cells. geogrid is the input geogrid
        with properties mapping cells for upscaling updated.
    """
    if any(r < 1 for r in refinement):
        raise ValueError(f"Refinement factors must be >= 1, got {refinement}")

    if region.dimensions != grid.dimensions:
        raise ValueError(
            f"Region property dimensions {region.dimensions} do not match "
            f"grid dimensions {grid.dimensions}"
        )

    # Make working copies so the caller's objects are not mutated.
    grid = grid.copy()
    region = region.copy()

    # Attach the region property to the grid.
    grid.append_prop(region)

    # 1. Crop to the bounding box of the target region.
    cropped, crop_origin = _crop_for_region(grid, region, target_region_id)

    # 2. Refine the cropped grid.
    refined = cropped.copy()
    rcol, rrow, rlay = refinement
    ocol, orow, olay = crop_origin
    refined.refine(refine_col=rcol, refine_row=rrow, refine_layer=rlay)
    _logger.info("Refined cropped grid dimensions: %s", refined.dimensions)

    # 3. Compute the NNC mapping table *before* deactivation mutates anything.
    #    This uses the original region property to find boundary faces and
    #    maps them through the crop → refine → merge index chain.
    nnc_table = _compute_nnc_table(
        region_prop=region,
        target_region_id=target_region_id,
        crop_origin=crop_origin,
        refinement=refinement,
        coarse_ncol=grid.ncol,
    )

    # 4. Deactivate the target region in the coarse grid (will be replaced).
    coarse_region = grid.get_prop_by_name(region.name)
    _set_actnum_by_region(grid, coarse_region, target_region_id, invert=False)

    # 5. In the refined grid keep only target-region cells active.
    refined_region = refined.get_prop_by_name(region.name)
    _set_actnum_by_region(refined, refined_region, target_region_id, invert=True)

    # 8. Merge the two grids.
    merged = xtgeo.grid_merge(grid, refined, olay, rlay)
    _logger.info("Merged grid dimensions: %s", merged.dimensions)

    if upscaling!=None:
        upscaling = _modify_upscaling_mapping(upscaling,
            region, target_region_id, refinement, crop_origin,
            ( refined.dimensions.ncol, refined.dimensions.nrow, refined.dimensions.nlay )
        )

  
    return merged, nnc_table, upscaling


def nnc_to_gridproperty(
    grid: xtgeo.Grid,
    nnc_df: pd.DataFrame,
) -> tuple[xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty]:
    """Convert NNC transmissibility data to three GridProperty instances.

    Takes the NNC DataFrame produced by :meth:`xtgeo.Grid.get_transmissibilities`
    and maps transmissibility values onto grid cells, producing one property per
    direction (I, J, K).

    For rows where DIRECTION contains ``"+"``, the transmissibility value is
    placed in cell ``(I1, J1, K1)``.  For rows where DIRECTION contains
    ``"-"``, the value is placed in cell ``(I2, J2, K2)``.  Index columns
    (I1, J1, K1, I2, J2, K2) are expected to be **1-based**.

    If multiple rows map to the same cell and direction, the transmissibility
    values are summed (parallel flow paths are additive).

    Args:
        grid: The xtgeo Grid that defines the geometry.
        nnc_df: A DataFrame with at least columns
            ``I1, J1, K1, I2, J2, K2, T, DIRECTION``.

    Returns:
        A tuple ``(tranx_nnc, trany_nnc, tranz_nnc)`` of
        :class:`xtgeo.GridProperty` instances named ``"TRANX_NNC"``,
        ``"TRANY_NNC"``, and ``"TRANZ_NNC"`` respectively.
        Cells without an NNC value are set to ``-1.0``.
    """
    required_cols = {"I1", "J1", "K1", "I2", "J2", "K2", "T", "DIRECTION"}
    missing = required_cols - set(nnc_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in nnc_df: {missing}")

    ncol, nrow, nlay = grid.ncol, grid.nrow, grid.nlay
    fill = -1.0

    arrays = {
        "I": np.zeros((ncol, nrow, nlay), dtype=np.float64),
        "J": np.zeros((ncol, nrow, nlay), dtype=np.float64),
        "K": np.zeros((ncol, nrow, nlay), dtype=np.float64),
    }
    touched = {
        "I": np.zeros((ncol, nrow, nlay), dtype=bool),
        "J": np.zeros((ncol, nrow, nlay), dtype=bool),
        "K": np.zeros((ncol, nrow, nlay), dtype=bool),
    }

    direction_col = nnc_df["DIRECTION"].astype(str)
    is_plus = direction_col.str.contains(r"\+", regex=True)
    is_minus = direction_col.str.contains("-")
    prefix_col = direction_col.str[0].str.upper()

    for prefix in ("I", "J", "K"):
        arr = arrays[prefix]
        tch = touched[prefix]
        mask_prefix = prefix_col == prefix

        # "+" rows → use (I1, J1, K1)
        sel_plus = nnc_df.loc[mask_prefix & is_plus]
        if not sel_plus.empty:
            ii = sel_plus["I1"].values.astype(int) - 1
            jj = sel_plus["J1"].values.astype(int) - 1
            kk = sel_plus["K1"].values.astype(int) - 1
            tt = sel_plus["T"].values.astype(float)
            valid = (
                (ii >= 0)
                & (ii < ncol)
                & (jj >= 0)
                & (jj < nrow)
                & (kk >= 0)
                & (kk < nlay)
            )
            np.add.at(arr, (ii[valid], jj[valid], kk[valid]), tt[valid])
            tch[ii[valid], jj[valid], kk[valid]] = True

        # "-" rows → use (I2, J2, K2)
        sel_minus = nnc_df.loc[mask_prefix & is_minus]
        if not sel_minus.empty:
            ii = sel_minus["I2"].values.astype(int) - 1
            jj = sel_minus["J2"].values.astype(int) - 1
            kk = sel_minus["K2"].values.astype(int) - 1
            tt = sel_minus["T"].values.astype(float)
            valid = (
                (ii >= 0)
                & (ii < ncol)
                & (jj >= 0)
                & (jj < nrow)
                & (kk >= 0)
                & (kk < nlay)
            )
            np.add.at(arr, (ii[valid], jj[valid], kk[valid]), tt[valid])
            tch[ii[valid], jj[valid], kk[valid]] = True

        # Set untouched cells to fill value
        arr[~tch] = fill

    prop_names = {"I": "TRANX_NNC", "J": "TRANY_NNC", "K": "TRANZ_NNC"}
    props = {}
    for prefix in ("I", "J", "K"):
        props[prefix] = xtgeo.GridProperty(
            grid,
            name=prop_names[prefix],
            values=np.ma.array(arrays[prefix]),
            discrete=False,
        )

    return props["I"], props["J"], props["K"]


def nnc_to_flowsimulator_input(
    nnc_df: pd.DataFrame,
    filepath: str | os.PathLike[str],
) -> None:
    """Write NNC transmissibilities to a flow-simulator input file.

    Produces a file with the ``NNC`` keyword suitable for reservoir
    simulators that use Eclipse-style input decks, such as Eclipse and
    OPM Flow.  The file can be included in the deck via ``INCLUDE``.
    Each row of *nnc_df* becomes one NNC record with the six cell
    indices and the transmissibility value.

    Args:
        nnc_df: A DataFrame with at least columns
            ``I1, J1, K1, I2, J2, K2, T``.  Optional columns ``TYPE``
            and ``DIRECTION`` are written as end-of-line comments.
        filepath: Path to the output file.
    """
    required_cols = {"I1", "J1", "K1", "I2", "J2", "K2", "T"}
    missing = required_cols - set(nnc_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in nnc_df: {missing}")

    has_type = "TYPE" in nnc_df.columns
    has_dir = "DIRECTION" in nnc_df.columns

    with open(filepath, "w") as f:
        f.write("NNC\n")
        for _, row in nnc_df.iterrows():
            line = (
                f"    {int(row['I1']):>4} {int(row['J1']):>4} {int(row['K1']):>4}"
                f"    {int(row['I2']):>4} {int(row['J2']):>4} {int(row['K2']):>4}"
                f"   {row['T']:.6f}  /"
            )
            comment_parts = []
            if has_type:
                comment_parts.append(str(row["TYPE"]))
            if has_dir:
                comment_parts.append(str(row["DIRECTION"]))
            if comment_parts:
                line += "  -- " + " ".join(comment_parts)
            f.write(line + "\n")
        f.write("/\n")

    _logger.info("NNC keyword written to %s", filepath)
