"""Zero-copy conversion between LGDO and Arrow tables.

Note on allocation: we use ``to_numpy(zero_copy_only=False)`` throughout.
PyArrow performs zero-copy for all numeric types and only allocates when it
must (e.g. booleans, which are bit-packed in Arrow but byte-packed in NumPy,
or columns containing nulls that need sentinel values). Multi-chunk columns
are combined automatically with a warning.
"""

import warnings

import numpy as np
import pyarrow as pa
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors
from lgdo.types.waveformtable import WaveformTable


# ============ LGDO → Arrow ============


def lgdo_to_arrow(lgdo_table: Table) -> pa.Table:
    """Convert LGDO Table to Arrow Table.

    - Nested Tables (e.g., WaveformTable) become StructArray columns
    - Preserves all attrs in Arrow field metadata
    - Uses native Arrow types (no Awkward extension types)

    Parameters
    ----------
    lgdo_table
        The LGDO Table to convert.

    Returns
    -------
    pa.Table
        Arrow table with native types and attrs as metadata.
    """
    struct_arr = _lgdo_col_to_arrow(lgdo_table)
    return pa.Table.from_batches([pa.RecordBatch.from_struct_array(struct_arr)])


def _lgdo_col_to_arrow(col) -> pa.Array:
    """Convert single LGDO column to Arrow array.

    Tables (including WaveformTable) become StructArrays with child field
    metadata preserving attrs like units.
    """
    if isinstance(col, Table):
        child_arrays = []
        child_fields = []
        for name, sub_col in col.items():
            child_arr = _lgdo_col_to_arrow(sub_col)
            meta = None
            if hasattr(sub_col, "attrs") and sub_col.attrs:
                meta = {k: str(v) for k, v in sub_col.attrs.items()}
            child_fields.append(pa.field(name, child_arr.type, metadata=meta))
            child_arrays.append(child_arr)
        return pa.StructArray.from_arrays(child_arrays, fields=child_fields)

    if isinstance(col, ArrayOfEqualSizedArrays):
        arr = pa.array(col.nda.ravel())
        for dim in reversed(col.nda.shape[1:]):
            arr = pa.FixedSizeListArray.from_arrays(arr, dim)
        return arr

    if isinstance(col, VectorOfVectors):
        return pa.ListArray.from_arrays(col._offsets.nda, _lgdo_col_to_arrow(col.flattened_data))

    if isinstance(col, Array):
        return pa.array(col.nda)

    raise TypeError(f"Unsupported LGDO type: {type(col)}")


# ============ Arrow → LGDO ============


def arrow_to_lgdo(arrow_table: pa.Table) -> Table:
    """Convert Arrow Table to LGDO Table.

    - Zero-copy where possible (requires single chunk, no nulls)
    - StructArray columns with fields {t0, dt, values} become WaveformTables
    - Restores units from Arrow field metadata

    Parameters
    ----------
    arrow_table
        The Arrow table to convert. Multi-chunk columns are combined
        automatically (with a warning, since this allocates).

    Returns
    -------
    Table
        LGDO Table with zero-copy views of Arrow data where possible.
    """
    col_dict = {}

    for name in arrow_table.column_names:
        field = arrow_table.schema.field(name)
        col = arrow_table.column(name)
        if col.num_chunks != 1:
            warnings.warn(
                f"Column '{name}' has {col.num_chunks} chunks; "
                "combining into one contiguous buffer (allocates memory)",
                stacklevel=2,
            )
        col_dict[name] = _arrow_col_to_lgdo(col.combine_chunks(), field)

    return Table(col_dict=col_dict)


def _arrow_col_to_lgdo(col: pa.Array, field: pa.Field | None):
    """Convert Arrow array to LGDO column (zero-copy).

    StructArrays whose fields are {t0, dt, values} become WaveformTables;
    other StructArrays become plain Tables.
    """
    attrs = (
        {k.decode(): v.decode() for k, v in field.metadata.items()}
        if field and field.metadata
        else None
    )

    if isinstance(col.type, pa.StructType):
        col_dict = {}
        for i in range(col.type.num_fields):
            sub_field = col.type.field(i)
            col_dict[sub_field.name] = _arrow_col_to_lgdo(col.field(sub_field.name), sub_field)

        if col_dict.keys() == {"t0", "dt", "values"}:
            # t0 and dt need to be writable as required by dspeed.build_processing_chain
            t0 = col_dict["t0"]
            dt = col_dict["dt"]
            return WaveformTable(
                t0=Array(nda=np.array(t0.nda, copy=True), attrs=t0.attrs),
                dt=Array(nda=np.array(dt.nda, copy=True), attrs=dt.attrs),
                values=col_dict["values"],
            )

        return Table(col_dict=col_dict)

    if isinstance(col.type, pa.FixedSizeListType):
        return ArrayOfEqualSizedArrays(nda=_nested_fixed_list_to_nda(col), attrs=attrs)

    if isinstance(col.type, pa.ListType):
        offsets = col.offsets.to_numpy(zero_copy_only=True, writable=False)

        if isinstance(col.values.type, pa.ListType):
            flattened = _arrow_col_to_lgdo(col.values, None)
        else:
            flattened = col.values.to_numpy(zero_copy_only=False, writable=False)

        return VectorOfVectors(flattened_data=flattened, offsets=offsets, attrs=attrs)

    return Array(nda=col.to_numpy(zero_copy_only=False, writable=False), attrs=attrs)


def _nested_fixed_list_to_nda(arr: pa.Array) -> np.ndarray:
    """Convert nested Arrow fixed_size_list to N-D numpy array."""
    dims = []
    while isinstance(arr.type, pa.FixedSizeListType):
        dims.append(arr.type.list_size)
        arr = arr.values
    return arr.to_numpy(zero_copy_only=False, writable=False).reshape(-1, *dims)
