"""A heavily cut down, modified and reformatted version of passagepipe.utils."""

import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike


@np.vectorize
def getObsIdFromQuery(obsName: str) -> int:
    """
    Cutout the obs ID from the long, jumbled MAST obs ID.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The observation ID.
    """

    return int(obsName.split("_")[0][7:-3])


@np.vectorize
def getExpIdFromQuery(obsName: str) -> int:
    """
    Cutout the exp ID from the long, jumbled MAST obs ID.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The exposure ID.
    """

    return int(obsName.split("_")[1])


def queryMAST(
    pid: int, nircam: bool = False, use_filter: ArrayLike | None = None
) -> Table:
    """
    Query MAST for the full list of observations for specific PID.

    Parameters
    ----------
    pid : int
        The JWST Proposal ID.
    nircam : bool, optional
        Select from NIRCam observations. By default `False`, only NIRISS
        observations will be returned.
    use_filter : ArrayLike | None, optional
        Return observations only in a specific set of filters, by default
        `None`.

    Returns
    -------
    Table
        The set of observations requested.
    """

    from astropy.table import vstack
    from astroquery.mast import Observations
    from mastquery import query

    query.DEFAULT_QUERY["project"] = ["JWST"]
    query.DEFAULT_QUERY["obs_collection"] = ["JWST"]
    if nircam:
        # query.DEFAULT_QUERY["instrument_name"] = ["NIRCAM/IMAGE", "NIRCAM/WFSS"]
        query.DEFAULT_QUERY["instrument_name"] = ["NIRCAM*"]
    else:
        # query.DEFAULT_QUERY["instrument_name"] = ["NIRISS/IMAGE*", "NIRISS/WFSS*", "NIRISS*"]
        query.DEFAULT_QUERY["instrument_name"] = ["NIRISS*"]

    queryList = query.run_query(
        box=None,
        proposal_id=[pid],
        base_query=query.DEFAULT_QUERY,
    )
    if use_filter is not None:
        queryList = queryList[np.isin(queryList["filter"], use_filter)]

    if "target_name" not in queryList.columns:
        queryList["target_name"] = queryList["target"]
    subqueryList = Observations.get_product_list(queryList)

    cond = (
        (subqueryList["calib_level"] == 1)
        # & (subqueryList["productType"] == "SCIENCE")
        & (subqueryList["productSubGroupDescription"] == "UNCAL")
    )

    uncalList = subqueryList[cond]
    _, idx = np.unique(uncalList["obs_id"], return_index=True)
    uncalList = uncalList[idx]

    uncalList["obs_id_num"] = getObsIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    uncalList["exp_id_num"] = getExpIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    return uncalList
