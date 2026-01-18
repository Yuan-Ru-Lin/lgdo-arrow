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

    - Flattens nested Tables (e.g., WaveformTable) with underscore prefixes
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

    def add_columns(table: Table, prefix: str = "") -> None:
        for name, col in table.items():
            full_name = f"{prefix}{name}" if prefix else name
            if isinstance(col, Table):
                add_columns(col, prefix=f"{full_name}_")
            else:
                arrays[full_name] = _lgdo_col_to_arrow(col)
                if hasattr(col, "attrs") and col.attrs:
                    fields_meta[full_name] = {k: str(v) for k, v in col.attrs.items()}

    add_columns(lgdo_table)

    # Build table and attach metadata to fields
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
    """Convert single LGDO column to Arrow array."""
    if isinstance(col, ArrayOfEqualSizedArrays):
        return _nda_to_nested_fixed_list(col.nda)

    if isinstance(col, VectorOfVectors):
        offsets = col._offsets.nda

        # Recurse if nested VectorOfVectors
        if isinstance(col.flattened_data, VectorOfVectors):
            values = _lgdo_col_to_arrow(col.flattened_data)
        else:
            values = pa.array(col.flattened_data.nda)

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
    - Reconstructs WaveformTable from *_t0, *_dt, *_values patterns
    - Restores units from Arrow field metadata

    Parameters
    ----------
    arrow_table
        The Arrow table to convert. Should have single chunk per column
        for zero-copy (use row_group_size=len(table) when writing Parquet).

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
    waveform_groups: dict[str, dict[str, str]] = {}

    for name in arrow_table.column_names:
        # Detect waveform column patterns
        for suffix in ("_t0", "_dt", "_values"):
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                waveform_groups.setdefault(prefix, {})[suffix[1:]] = name
                break
        else:
            field = arrow_table.schema.field(name)
            chunk = _get_single_chunk(arrow_table.column(name))
            col_dict[name] = _arrow_col_to_lgdo(chunk, field)

    # Reconstruct WaveformTables
    for prefix, parts in waveform_groups.items():
        if set(parts.keys()) == {"t0", "dt", "values"}:
            t0_field = arrow_table.schema.field(parts["t0"])
            dt_field = arrow_table.schema.field(parts["dt"])
            values_field = arrow_table.schema.field(parts["values"])

            # `t0` and `dt` need to be writable as required by `dspeed.build_processing_chain`.
            t0 = Array(
                nda=_get_single_chunk(arrow_table.column(parts["t0"])).to_numpy(zero_copy_only=False, writable=True),
                attrs=_extract_attrs(t0_field) if t0_field else None,
            )
            dt = Array(
                nda=_get_single_chunk(arrow_table.column(parts["dt"])).to_numpy(zero_copy_only=False, writable=True),
                attrs=_extract_attrs(dt_field) if dt_field else None,
            )
            values = _arrow_col_to_lgdo(
                _get_single_chunk(arrow_table.column(parts["values"])), values_field
            )

            col_dict[prefix] = WaveformTable(t0=t0, dt=dt, values=values)

    return Table(col_dict=col_dict)


def _get_single_chunk(col: pa.ChunkedArray) -> pa.Array:
    """Get single chunk from ChunkedArray; error if multiple."""
    if col.num_chunks != 1:
        raise ValueError(
            f"Expected 1 chunk, got {col.num_chunks}. "
            "Write Parquet with row_group_size=len(table) for zero-copy."
        )
    return col.chunk(0)


def _arrow_col_to_lgdo(col: pa.Array, field: pa.Field | None):
    """Convert Arrow array to LGDO column (zero-copy)."""
    attrs = _extract_attrs(field) if field else None

    if isinstance(col.type, pa.FixedSizeListType):
        nda = _nested_fixed_list_to_nda(col)
        return ArrayOfEqualSizedArrays(nda=nda, attrs=attrs)

    if isinstance(col.type, pa.ListType):
        offsets = col.offsets.to_numpy(zero_copy_only=True, writable=False)

        # Recurse if nested list
        if isinstance(col.values.type, pa.ListType):
            flattened = _arrow_col_to_lgdo(col.values, None)
        else:
            flattened = col.values.to_numpy(zero_copy_only=True, writable=False)

        return VectorOfVectors(
            flattened_data=flattened,
            cumulative_length=offsets[1:],
            attrs=attrs,
        )

    nda = col.to_numpy(zero_copy_only=True, writable=False)
    return Array(nda=nda, attrs=attrs)


def _nested_fixed_list_to_nda(arr: pa.Array) -> np.ndarray:
    """Convert nested Arrow fixed_size_list to N-D numpy array (zero-copy)."""
    dims = []

    while isinstance(arr.type, pa.FixedSizeListType):
        dims.append(arr.type.list_size)
        arr = arr.values

    flat = arr.to_numpy(zero_copy_only=True, writable=False)
    return flat.reshape(-1, *dims)


def _extract_attrs(field: pa.Field) -> dict | None:
    """Extract all attrs from Arrow field metadata."""
    if field.metadata:
        return {k.decode(): v.decode() for k, v in field.metadata.items()}
    return None
