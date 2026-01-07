"""Zero-copy conversion between LGDO and Arrow tables."""

import numpy as np
import pyarrow as pa
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors
from lgdo.types.waveformtable import WaveformTable


# ============ LGDO → Arrow ============


def lgdo_to_arrow(lgdo_table: Table) -> pa.Table:
    """Convert LGDO Table to Arrow Table.

    - Flattens nested Tables (e.g., WaveformTable) with underscore prefixes
    - Preserves units in Arrow field metadata
    - Uses native Arrow types (no Awkward extension types)

    Parameters
    ----------
    lgdo_table
        The LGDO Table to convert.

    Returns
    -------
    pa.Table
        Arrow table with native types and unit metadata.
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
                units = col.attrs.get("units") if hasattr(col, "attrs") else None
                if units:
                    fields_meta[full_name] = {"units": str(units)}

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
        nda = col.nda
        flat = pa.array(nda.ravel())
        return pa.FixedSizeListArray.from_arrays(flat, nda.shape[1])

    if isinstance(col, VectorOfVectors):
        offsets = np.concatenate([[0], col.cumulative_length])
        values = pa.array(col.flattened_data)
        return pa.ListArray.from_arrays(pa.array(offsets), values)

    if isinstance(col, Array):
        return pa.array(col.nda)

    raise TypeError(f"Unsupported LGDO type: {type(col)}")


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

            t0 = _arrow_col_to_lgdo(
                _get_single_chunk(arrow_table.column(parts["t0"])), t0_field
            )
            dt = _arrow_col_to_lgdo(
                _get_single_chunk(arrow_table.column(parts["dt"])), dt_field
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


def _arrow_col_to_lgdo(col: pa.Array, field: pa.Field):
    """Convert Arrow array to LGDO column (zero-copy)."""
    attrs = _extract_units(field)

    if isinstance(col.type, pa.FixedSizeListType):
        flat = col.values.to_numpy(zero_copy_only=True, writable=False)
        nda = flat.reshape(-1, col.type.list_size)
        return ArrayOfEqualSizedArrays(nda=nda, attrs=attrs)

    if isinstance(col.type, pa.ListType):
        offsets = col.offsets.to_numpy(zero_copy_only=True, writable=False)
        values = col.values.to_numpy(zero_copy_only=True, writable=False)
        return VectorOfVectors(
            flattened_data=values,
            cumulative_length=offsets[1:],
            attrs=attrs,
        )

    nda = col.to_numpy(zero_copy_only=True, writable=False)
    return Array(nda=nda, attrs=attrs)


def _extract_units(field: pa.Field) -> dict | None:
    """Extract units from Arrow field metadata."""
    if field.metadata and b"units" in field.metadata:
        return {"units": field.metadata[b"units"].decode()}
    return None
