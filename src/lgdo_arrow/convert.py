"""Zero-copy conversion between LGDO and Arrow tables."""

import numpy as np
import pyarrow as pa
# `VectorOfEncodedVectors`, `ArrayOfEncodedEqualSizedArrays` are not handled
# here since enconding and decoding are handled by libraries like pyarrow
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
    arrays = {}
    fields_meta = {}

    for name, col in lgdo_table.items():
        arrays[name] = _lgdo_col_to_arrow(col)
        # Top-level leaf columns carry attrs on the pa.Table field.
        # For Tables, attrs are already embedded in the StructArray child fields.
        if not isinstance(col, Table) and hasattr(col, "attrs") and col.attrs:
            fields_meta[name] = {k: str(v) for k, v in col.attrs.items()}

    table = pa.table(arrays)
    if fields_meta:
        new_fields = []
        for field in table.schema:
            meta = fields_meta.get(field.name)
            if meta:
                field = field.with_metadata(meta)
            new_fields.append(field)
        table = table.cast(pa.schema(new_fields))

    return table


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
        return _nda_to_nested_fixed_list(col.nda)

    if isinstance(col, VectorOfVectors):
        offsets = col._offsets.nda
        values = _lgdo_col_to_arrow(col.flattened_data)
        return pa.ListArray.from_arrays(offsets, values)

    if isinstance(col, Array):
        return pa.array(col.nda)

    raise TypeError(f"Unsupported LGDO type: {type(col)}")


def _nda_to_nested_fixed_list(nda: np.ndarray) -> pa.Array:
    """Convert N-D numpy array to nested Arrow fixed_size_list."""
    arr = pa.array(nda.ravel())

    # Wrap dimensions from innermost to outermost
    for dim in reversed(nda.shape[1:]):
        arr = pa.FixedSizeListArray.from_arrays(arr, dim)

    return arr


# ============ Arrow → LGDO ============


def arrow_to_lgdo(arrow_table: pa.Table) -> Table:
    """Convert Arrow Table to LGDO Table.

    - Zero-copy where possible (requires single chunk, no nulls)
    - StructArray columns with fields {t0, dt, values} become WaveformTables
    - Restores units from Arrow field metadata

    Parameters
    ----------
    arrow_table
        The Arrow table to convert. Must have single chunk per column
        for zero-copy. Use ``table.combine_chunks()`` if needed.

    Returns
    -------
    Table
        LGDO Table with zero-copy views of Arrow data.

    Raises
    ------
    ValueError
        If a column has multiple chunks.
    pyarrow.ArrowInvalid
        If zero-copy is not possible (nulls, incompatible types).
    """
    col_dict = {}

    for name in arrow_table.column_names:
        field = arrow_table.schema.field(name)
        chunk = _get_single_chunk(arrow_table.column(name))
        col_dict[name] = _arrow_col_to_lgdo(chunk, field)

    return Table(col_dict=col_dict)


def _to_numpy_zero_copy_except_bool(arr: pa.Array) -> np.ndarray:
    """Convert Arrow array to numpy, zero-copy except for booleans.

    Arrow booleans are bit-packed (1 bit each) while NumPy uses 1 byte each,
    so zero-copy is impossible for boolean arrays.
    """
    is_bool = pa.types.is_boolean(arr.type)
    return arr.to_numpy(zero_copy_only=not is_bool, writable=False)


def _get_single_chunk(col: pa.ChunkedArray) -> pa.Array:
    """Get single chunk from ChunkedArray; error if multiple."""
    if col.num_chunks != 1:
        raise ValueError(
            f"Expected 1 chunk, got {col.num_chunks}. "
            "Use table.combine_chunks() before converting to LGDO."
        )
    return col.chunk(0)


def _arrow_col_to_lgdo(col: pa.Array, field: pa.Field | None):
    """Convert Arrow array to LGDO column (zero-copy).

    StructArrays whose fields are {t0, dt, values} become WaveformTables;
    other StructArrays become plain Tables.
    """
    attrs = _extract_attrs(field) if field else None

    if isinstance(col.type, pa.StructType):
        field_names = {col.type.field(i).name for i in range(col.type.num_fields)}
        col_dict = {}
        for i in range(col.type.num_fields):
            sub_field = col.type.field(i)
            col_dict[sub_field.name] = _arrow_col_to_lgdo(col.field(sub_field.name), sub_field)

        if field_names == {"t0", "dt", "values"}:
            # t0 and dt need to be writable as required by dspeed.build_processing_chain
            t0 = col_dict["t0"]
            dt = col_dict["dt"]
            t0 = Array(nda=np.array(t0.nda, copy=True), attrs=t0.attrs)
            dt = Array(nda=np.array(dt.nda, copy=True), attrs=dt.attrs)
            return WaveformTable(t0=t0, dt=dt, values=col_dict["values"])

        return Table(col_dict=col_dict)

    if isinstance(col.type, pa.FixedSizeListType):
        nda = _nested_fixed_list_to_nda(col)
        return ArrayOfEqualSizedArrays(nda=nda, attrs=attrs)

    if isinstance(col.type, pa.ListType):
        offsets = col.offsets.to_numpy(zero_copy_only=True, writable=False)

        # Recurse if nested list
        if isinstance(col.values.type, pa.ListType):
            flattened = _arrow_col_to_lgdo(col.values, None)
        else:
            flattened = _to_numpy_zero_copy_except_bool(col.values)

        return VectorOfVectors(
            flattened_data=flattened,
            offsets=offsets,
            attrs=attrs,
        )

    nda = _to_numpy_zero_copy_except_bool(col)
    return Array(nda=nda, attrs=attrs)


def _nested_fixed_list_to_nda(arr: pa.Array) -> np.ndarray:
    """Convert nested Arrow fixed_size_list to N-D numpy array (zero-copy)."""
    dims = []

    while isinstance(arr.type, pa.FixedSizeListType):
        dims.append(arr.type.list_size)
        arr = arr.values

    flat = _to_numpy_zero_copy_except_bool(arr)
    return flat.reshape(-1, *dims)


def _extract_attrs(field: pa.Field) -> dict | None:
    """Extract all attrs from Arrow field metadata."""
    if field.metadata:
        return {k.decode(): v.decode() for k, v in field.metadata.items()}
    return None
