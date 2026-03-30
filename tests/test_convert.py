"""Tests for lgdo_arrow.convert round-trip conversions."""

import numpy as np
import pyarrow as pa
import pytest
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors
from lgdo.types.waveformtable import WaveformTable
from numpy.testing import assert_array_equal

from lgdo_arrow import arrow_to_lgdo, lgdo_to_arrow


# ============ lgdo_to_arrow output types ============


class TestLgdoToArrowTypes:
    def test_table_returns_pa_table(self):
        tbl = Table(col_dict={"x": Array(nda=np.array([1, 2]))})
        assert isinstance(lgdo_to_arrow(tbl), pa.Table)

    def test_waveformtable_returns_struct_array(self):
        wft = _make_waveform_table()
        result = lgdo_to_arrow(wft)
        assert isinstance(result, pa.StructArray)
        assert isinstance(result.type, pa.StructType)

    def test_array_returns_pa_array(self):
        result = lgdo_to_arrow(Array(nda=np.array([1.0, 2.0])))
        assert isinstance(result, pa.Array)
        assert not isinstance(result, pa.StructArray)

    def test_aoesa_returns_fixed_size_list(self):
        result = lgdo_to_arrow(ArrayOfEqualSizedArrays(nda=np.zeros((3, 4))))
        assert isinstance(result.type, pa.FixedSizeListType)

    def test_vov_returns_list_array(self):
        vov = VectorOfVectors(
            flattened_data=np.array([1, 2, 3]),
            offsets=np.array([0, 2, 3]),
        )
        assert isinstance(lgdo_to_arrow(vov), pa.ListArray)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported LGDO type"):
            lgdo_to_arrow("not an lgdo object")


# ============ arrow_to_lgdo output types ============


class TestArrowToLgdoTypes:
    def test_pa_table_returns_table(self):
        tbl = pa.table({"x": [1, 2, 3]})
        assert isinstance(arrow_to_lgdo(tbl), Table)

    def test_struct_array_with_waveform_fields_returns_waveformtable(self):
        wft = _make_waveform_table()
        struct = lgdo_to_arrow(wft)
        assert isinstance(arrow_to_lgdo(struct), WaveformTable)

    def test_struct_array_without_waveform_fields_returns_table(self):
        struct = pa.StructArray.from_arrays(
            [pa.array([1, 2]), pa.array([3, 4])],
            names=["a", "b"],
        )
        result = arrow_to_lgdo(struct)
        assert isinstance(result, Table)
        assert not isinstance(result, WaveformTable)

    def test_fixed_size_list_returns_aoesa(self):
        arr = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(12)), 4)
        assert isinstance(arrow_to_lgdo(arr), ArrayOfEqualSizedArrays)

    def test_list_array_returns_vov(self):
        arr = pa.ListArray.from_arrays(
            pa.array([0, 2, 3, 5]),
            pa.array([1, 2, 3, 4, 5]),
        )
        assert isinstance(arrow_to_lgdo(arr), VectorOfVectors)

    def test_primitive_array_returns_array(self):
        assert isinstance(arrow_to_lgdo(pa.array([1, 2, 3])), Array)

    def test_chunked_array(self):
        chunked = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
        with pytest.warns(UserWarning, match="2 chunks"):
            result = arrow_to_lgdo(chunked)
        assert isinstance(result, Array)
        assert_array_equal(result.nda, [1, 2, 3, 4])

    def test_single_chunk_no_warning(self):
        chunked = pa.chunked_array([pa.array([1, 2, 3])])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = arrow_to_lgdo(chunked)
        assert isinstance(result, Array)


# ============ Round-trip data integrity ============


class TestRoundTrip:
    def test_array(self):
        original = Array(nda=np.array([1.5, 2.5, 3.5]))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, Array)
        assert_array_equal(back.nda, original.nda)

    def test_array_int(self):
        original = Array(nda=np.array([10, 20, 30], dtype=np.int32))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert_array_equal(back.nda, original.nda)
        assert back.nda.dtype == np.int32

    def test_aoesa_2d(self):
        original = ArrayOfEqualSizedArrays(nda=np.arange(12).reshape(3, 4))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, ArrayOfEqualSizedArrays)
        assert_array_equal(back.nda, original.nda)

    def test_aoesa_3d(self):
        original = ArrayOfEqualSizedArrays(nda=np.arange(24).reshape(2, 3, 4))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert_array_equal(back.nda, original.nda)

    def test_vov(self):
        flat = np.array([10, 20, 30, 40, 50])
        offsets = np.array([0, 2, 3, 5])
        original = VectorOfVectors(flattened_data=flat, offsets=offsets)
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, VectorOfVectors)
        assert_array_equal(back.flattened_data, flat)
        assert_array_equal(back._offsets.nda, offsets)

    def test_vov_with_aoesa_flattened_data(self):
        """VoV whose flattened_data is an AOESA (ListArray<FixedSizeListArray>)."""
        inner = ArrayOfEqualSizedArrays(nda=np.arange(12).reshape(4, 3))
        offsets = np.array([0, 2, 4])
        original = VectorOfVectors(flattened_data=inner, offsets=offsets)
        arrow = lgdo_to_arrow(original)
        back = arrow_to_lgdo(arrow)
        assert isinstance(back, VectorOfVectors)
        assert isinstance(back.flattened_data, ArrayOfEqualSizedArrays)
        assert_array_equal(back.flattened_data.nda, inner.nda)
        assert_array_equal(back._offsets.nda, offsets)

    def test_waveform_table(self):
        original = _make_waveform_table()
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, WaveformTable)
        assert_array_equal(back["t0"].nda, original["t0"].nda)
        assert_array_equal(back["dt"].nda, original["dt"].nda)
        assert_array_equal(back["values"].nda, original["values"].nda)

    def test_table(self):
        original = Table(col_dict={
            "energy": Array(nda=np.array([1.0, 2.0, 3.0])),
            "channel": Array(nda=np.array([0, 1, 2])),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, Table)
        assert_array_equal(back["energy"].nda, original["energy"].nda)
        assert_array_equal(back["channel"].nda, original["channel"].nda)

    def test_table_with_nested_waveform(self):
        original = Table(col_dict={
            "energy": Array(nda=np.array([1.0, 2.0])),
            "waveform": _make_waveform_table(),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back["waveform"], WaveformTable)
        assert_array_equal(
            back["waveform"]["values"].nda,
            original["waveform"]["values"].nda,
        )


# ============ Attrs round-trip ============


class TestAttrsRoundTrip:
    def test_string_attr(self):
        tbl = Table(col_dict={
            "x": Array(nda=np.array([1.0]), attrs={"units": "keV"}),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["units"] == "keV"

    def test_numeric_attr(self):
        tbl = Table(col_dict={
            "x": Array(nda=np.array([1.0]), attrs={"version": 3, "scale": 1.5}),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["version"] == 3
        assert isinstance(back["x"].attrs["version"], int)
        assert back["x"].attrs["scale"] == 1.5

    def test_bool_attr(self):
        tbl = Table(col_dict={
            "x": Array(nda=np.array([1.0]), attrs={"calibrated": True}),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["calibrated"] is True

    def test_dict_attr(self):
        tbl = Table(col_dict={
            "x": Array(nda=np.array([1.0]), attrs={"info": {"a": 1, "b": 2}}),
        })
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["info"] == {"a": 1, "b": 2}

    def test_waveform_attrs(self):
        wft = _make_waveform_table()
        tbl = Table(col_dict={"wf": wft})
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["wf"]["t0"].attrs["units"] == "us"
        assert back["wf"]["dt"].attrs["units"] == "ns"


# ============ Helpers ============


import warnings


def _make_waveform_table():
    return WaveformTable(
        t0=Array(nda=np.array([0.0, 0.1]), attrs={"units": "us"}),
        dt=Array(nda=np.array([16.0, 16.0]), attrs={"units": "ns"}),
        values=ArrayOfEqualSizedArrays(
            nda=np.arange(10, dtype=np.float32).reshape(2, 5)
        ),
    )
