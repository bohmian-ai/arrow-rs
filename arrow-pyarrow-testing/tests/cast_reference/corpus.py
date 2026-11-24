# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""The Arrow C++ cast test corpus.

Translated from `cpp/src/arrow/compute/kernels/scalar_cast_test.cc` of Apache
Arrow 24.0.0 (tag `apache-arrow-24.0.0`, commit
`31b4b6c0a0a7e7c117312d285541a21446675ec6`), licensed to the Apache Software
Foundation under the Apache License, Version 2.0. Each `@cpp_test` function
translates the C++ test of the same name; its cases keep the C++ inputs,
options and expected results or refusals.

`check` is C++ `CheckCast`, `fails` is `CheckCastFails`, `unsupported` is an
expected `NotImplemented` refusal and `attempt` compares the engines without a
C++ expectation. A C++ test whose assertions cannot be expressed as a cast
outcome is listed in `MAPPED_ELSEWHERE` with the Rust test that covers it.
"""

import decimal
import json

import pyarrow as pa

# Every TEST in scalar_cast_test.cc at the pinned commit, in source order.
CPP_TESTS = """
CanCast SameTypeZeroCopy ZeroChunks CastDoesNotProvideDefaultOptions FromBoolean
ToBoolean ToIntUpcast OverflowInNullSlot ToIntDowncastSafe IntegerSignedToUnsigned
IntegerUnsignedToSigned ToIntDowncastUnsafe FloatingToInt FloatingToFloating
IntToFloating Decimal32ToInt Decimal64ToInt Decimal128ToInt Decimal256ToInt
IntegerToDecimal Decimal32ToDecimal32 Decimal64ToDecimal64 Decimal128ToDecimal128
Decimal256ToDecimal256 Decimal32ToDecimal64 Decimal32ToDecimal128
Decimal32ToDecimal256 Decimal64ToDecimal32 Decimal64ToDecimal128
Decimal64ToDecimal256 Decimal128ToDecimal32 Decimal128ToDecimal64
Decimal128ToDecimal256 Decimal256ToDecimal32 Decimal256ToDecimal64
Decimal256ToDecimal128 FloatingToDecimal DecimalToFloating DecimalToString
TimestampToTimestamp TimestampZeroCopy TimestampToTimestampMultiplyOverflow
TimestampToDate ZonedTimestampToDate TimestampToTime ZonedTimestampToTime
TimeToTime TimeZeroCopy DateToString TimeToString TimestampToString
TimestampWithZoneToString DateToDate DateZeroCopy DurationToDuration
DurationZeroCopy DurationToDurationMultiplyOverflow DurationToString
MiscToFloating UnsupportedInputType UnsupportedTargetType StringToBoolean
StringToInt StringToFloating StringToDecimal StringToTimestamp StringToDate
BinaryToString BinaryOrStringToBinary StringToString StringToStringWithOffset
BinaryOrStringToFixedSizeBinary FixedSizeBinaryToBinaryOrString
FixedSizeBinaryToBinaryOrStringWithSlice IntToString FloatingToString
BooleanToString ListToPrimitive ListToList ListToListNoNulls
ListToListOptionsPassthru FSLToFSL FSLToFSLNoNulls FSLToFSLOptionsPassThru
FSLToList ListToFSL CastMap StructToSameSizedAndNamedStruct
StructToStructSubset StructToStructSubsetWithNulls
StructToSameSizedButDifferentNamedStruct StructToBiggerStruct
StructToBiggerNullableStruct StructToDifferentNullabilityStruct IdentityCasts
EmptyCasts CastWithNoValidityBitmapButUnknownNullCount FromNull
FromNullToDictionary FromDictionary ExtensionTypeToIntDowncast
PrimitiveToExtension ExtensionDictToExtension IntToExtensionTypeDowncast
DictTypeToAnotherDict NoOutBitmapIfInIsAllValid
""".split()

# C++ tests whose assertions are not cast outcomes, and the Rust test owning
# the equivalent assertion.
MAPPED_ELSEWHERE = {}

UNSAFE = dict(
    allow_int_overflow=True,
    allow_time_truncate=True,
    allow_time_overflow=True,
    allow_decimal_truncate=True,
    allow_float_truncate=True,
    allow_invalid_utf8=True,
)


class Case:
    """One cast of `input` to `to` under `options`, expecting `expect`: an
    array, a status category, `"supported"` (any outcome but
    `NotImplemented`) or `None` (no C++ expectation)."""

    def __init__(self, input, to, options, expect, label, variants=True):
        self.input = input
        self.to = to
        self.options = dict(options)
        self.expect = expect
        self.label = label
        self.variants = variants


def check(input, expected, label="", **options):
    return Case(input, expected.type, options, expected, label or "check")


def fails(input, to, label="", **options):
    return Case(input, to, options, "Invalid", label or "fails")


def unsupported(input, to, label=""):
    return Case(input, to, {}, "NotImplemented", label or "unsupported")


def attempt(input, to, label="", **options):
    return Case(input, to, options, None, label or "attempt")


_REGISTRY = []


def cpp_test(name):
    def register(function):
        _REGISTRY.append((name, function))
        return function

    return register


def _value(type, value):
    """A Python value for one C++ ArrayFromJSON element of `type`."""
    if value is None:
        return None
    if pa.types.is_decimal(type):
        return decimal.Decimal(value)
    if pa.types.is_floating(type) and type != pa.float16():
        return float(value)
    if pa.types.is_binary(type) or pa.types.is_large_binary(type) or pa.types.is_fixed_size_binary(type) or pa.types.is_binary_view(type):
        return value.encode("utf-8", "surrogateescape") if isinstance(value, str) else value
    if pa.types.is_list(type) or pa.types.is_large_list(type) or pa.types.is_fixed_size_list(type) or pa.types.is_list_view(type) or pa.types.is_large_list_view(type):
        return [_value(type.value_type, v) for v in value]
    if pa.types.is_map(type):
        return [(_value(type.key_type, k), _value(type.item_type, v)) for k, v in value]
    if pa.types.is_struct(type):
        if isinstance(value, list):
            value = {type.field(i).name: v for i, v in enumerate(value)}
        return {
            type.field(i).name: _value(type.field(i).type, value.get(type.field(i).name))
            for i in range(type.num_fields)
        }
    return value


def J(type, text):
    """C++ `ArrayFromJSON(type, text)`."""
    values = json.loads(text) if isinstance(text, str) else text
    if pa.types.is_timestamp(type) and any(isinstance(v, str) for v in values):
        return pa.array(values, pa.string()).cast(type)
    if pa.types.is_dictionary(type):
        return pa.array([_value(type.value_type, v) for v in values], type.value_type).dictionary_encode().cast(type)
    return pa.array([_value(type, v) for v in values], type)


def D(type, indices, dictionary):
    """C++ `DictArrayFromJSON(type, indices, dictionary)`."""
    return pa.DictionaryArray.from_arrays(J(type.index_type, indices), J(type.value_type, dictionary))


def mask_nulls_at(array, indices):
    """C++ `MaskArrayWithNullsAt`: the same buffers with `indices` made null."""
    valid = [array[i].is_valid and i not in indices for i in range(len(array))]
    bitmap = pa.array(valid, pa.bool_()).buffers()[1]
    buffers = array.buffers()
    return pa.Array.from_buffers(array.type, len(array), [bitmap] + buffers[1:], offset=0) if array.offset == 0 else None


def decimal_unscaled(type, unscaled):
    """A decimal array of `type` holding the unscaled integers `unscaled`."""
    width = type.byte_width
    data = b"".join(
        b"\0" * width if v is None else int(v).to_bytes(width, "little", signed=True)
        for v in unscaled
    )
    validity = pa.array([v is not None for v in unscaled], pa.bool_()).buffers()[1]
    return pa.Array.from_buffers(type, len(unscaled), [validity, pa.py_buffer(data)])


NUMERIC = [pa.uint8(), pa.int8(), pa.uint16(), pa.int16(), pa.uint32(), pa.int32(), pa.uint64(), pa.int64(), pa.float32(), pa.float64()]
INTEGER = [pa.int8(), pa.uint8(), pa.int16(), pa.uint16(), pa.int32(), pa.uint32(), pa.int64(), pa.uint64()]
BASE_BINARY = [pa.binary(), pa.utf8(), pa.large_binary(), pa.large_utf8()]
BASE_BINARY_AND_VIEW = BASE_BINARY + [pa.string_view(), pa.binary_view()]


def can_cast(from_type, to_types, expected=True):
    """C++ `ExpectCanCast`: the cast of an all-null array is supported."""
    for to in to_types:
        input = pa.nulls(2, from_type)
        expect = "supported" if expected else "NotImplemented"
        label = f"CanCast({from_type}, {to})={expected}"
        if pa.types.is_dictionary(from_type) and (
            pa.types.is_string_view(from_type.value_type) or pa.types.is_binary_view(from_type.value_type)
        ):
            # Registered, so C++ CanCast is true, but its execution has no
            # take kernel for view values (GH-43010).
            expect = "NotImplemented"
            label += " executes as NotImplemented"
        yield Case(input, to, {}, expect, label, variants=False)


@cpp_test("CanCast")
def _():
    yield from can_cast(pa.null(), [pa.bool_()])
    yield from can_cast(pa.null(), NUMERIC)
    yield from can_cast(pa.null(), BASE_BINARY_AND_VIEW)
    yield from can_cast(pa.null(), [pa.date32(), pa.date64(), pa.time32("ms"), pa.timestamp("s")])
    yield from can_cast(pa.dictionary(pa.uint16(), pa.null()), [pa.null()])

    yield from can_cast(pa.bool_(), [pa.bool_()])
    yield from can_cast(pa.bool_(), NUMERIC)
    yield from can_cast(pa.bool_(), [pa.utf8(), pa.string_view(), pa.large_utf8()])
    yield from can_cast(pa.dictionary(pa.int32(), pa.bool_()), [pa.bool_()])

    yield from can_cast(pa.bool_(), [pa.null()], False)
    yield from can_cast(pa.bool_(), [pa.binary(), pa.large_binary()], False)
    yield from can_cast(pa.bool_(), [pa.date32(), pa.date64(), pa.time32("ms"), pa.timestamp("s")], False)

    for from_numeric in NUMERIC:
        yield from can_cast(from_numeric, [pa.bool_()])
        yield from can_cast(from_numeric, NUMERIC)
        yield from can_cast(from_numeric, [pa.utf8(), pa.large_utf8()])
        yield from can_cast(pa.dictionary(pa.int32(), from_numeric), [from_numeric])
        yield from can_cast(from_numeric, [pa.null()], False)

    for from_base_binary in BASE_BINARY_AND_VIEW:
        yield from can_cast(from_base_binary, [pa.bool_()])
        yield from can_cast(from_base_binary, NUMERIC)
        yield from can_cast(from_base_binary, BASE_BINARY)
        if from_base_binary not in (pa.string_view(), pa.binary_view()):
            yield from can_cast(pa.dictionary(pa.int64(), from_base_binary), [from_base_binary])
        yield from can_cast(pa.dictionary(pa.uint32(), from_base_binary), BASE_BINARY)
        yield from can_cast(pa.dictionary(pa.int16(), from_base_binary), NUMERIC)
        yield from can_cast(from_base_binary, [pa.null()], False)

    yield from can_cast(pa.utf8(), [pa.timestamp("ms"), pa.date32(), pa.date64()])
    yield from can_cast(pa.large_utf8(), [pa.timestamp("ns"), pa.date32(), pa.date64()])
    yield from can_cast(pa.timestamp("us"), [pa.binary(), pa.large_binary()], False)

    yield from can_cast(pa.binary(3), BASE_BINARY_AND_VIEW)
    yield from can_cast(pa.binary(3), [pa.binary(3)])

    # smallint and tinyint are C++ test extension types over int16 and int8.
    yield from can_cast(pa.int16(), [pa.int16()], True)
    yield from can_cast(pa.int16(), NUMERIC)

    yield from can_cast(pa.date32(), [pa.utf8(), pa.large_utf8()])
    yield from can_cast(pa.date64(), [pa.utf8(), pa.large_utf8()])
    yield from can_cast(pa.timestamp("ns"), [pa.utf8(), pa.large_utf8()])
    yield from can_cast(pa.timestamp("us"), [pa.utf8(), pa.large_utf8()])
    yield from can_cast(pa.time32("ms"), [pa.utf8(), pa.large_utf8()])
    yield from can_cast(pa.time64("ns"), [pa.utf8(), pa.large_utf8()])


@cpp_test("ZeroChunks")
def _():
    # A chunked array of no chunks casts to an empty result; its Rust form is
    # an empty array.
    yield check(pa.array([], pa.int32()), pa.array([], pa.utf8()))


@cpp_test("FromBoolean")
def _():
    vals = "[1, 0, null, 1, 0, 1, 1, null, 0, 0, 1]"
    yield check(J(pa.bool_(), [None if v is None else bool(v) for v in json.loads(vals)]), J(pa.int32(), vals))


@cpp_test("ToBoolean")
def _():
    for type in NUMERIC:
        yield check(J(type, "[0, null, 127, 1, 0]"), J(pa.bool_(), "[false, null, true, true, false]"))
    for type in [pa.int8(), pa.float64()]:
        yield check(J(type, "[0, null, 127, -1, 0]"), J(pa.bool_(), "[false, null, true, true, false]"))


@cpp_test("ToIntUpcast")
def _():
    yield check(J(pa.int8(), "[0, null, 127, -1, 0]"), J(pa.int32(), "[0, null, 127, -1, 0]"))
    yield check(J(pa.uint8(), "[0, 100, 200, 255, 0]"), J(pa.int16(), "[0, 100, 200, 255, 0]"))


@cpp_test("OverflowInNullSlot")
def _():
    yield check(
        mask_nulls_at(J(pa.int32(), "[0, 87654321, 2000, 1000, 0]"), [1]),
        J(pa.int16(), "[0, null, 2000, 1000, 0]"),
    )


@cpp_test("ToIntDowncastSafe")
def _():
    yield check(J(pa.int16(), "[0, null, 200, 1, 2]"), J(pa.uint8(), "[0, null, 200, 1, 2]"))
    yield fails(J(pa.int16(), "[0, null, 256, 0, 0]"), pa.uint8())
    yield fails(J(pa.int16(), "[0, null, -1, 0, 0]"), pa.uint8())
    yield check(J(pa.int32(), "[0, null, 2000, 1, 2]"), J(pa.int16(), "[0, null, 2000, 1, 2]"))
    yield fails(J(pa.int32(), "[0, null, 2000, 70000, 2]"), pa.int16())
    yield fails(J(pa.int32(), "[0, null, 2000, -70000, 2]"), pa.int16())
    yield fails(J(pa.int32(), "[0, null, 2000, -70000, 2]"), pa.uint8())


@cpp_test("IntegerSignedToUnsigned")
def _():
    i32s = J(pa.int32(), "[-2147483648, null, -1, 65535, 2147483647]")
    yield fails(i32s, pa.uint32())
    yield fails(i32s, pa.uint64())
    yield fails(i32s, pa.uint16())
    yield check(i32s, J(pa.uint32(), "[2147483648, null, 4294967295, 65535, 2147483647]"), allow_int_overflow=True)
    yield check(i32s, J(pa.uint64(), "[18446744071562067968, null, 18446744073709551615, 65535, 2147483647]"), allow_int_overflow=True)
    yield check(i32s, J(pa.uint16(), "[0, null, 65535, 65535, 65535]"), allow_int_overflow=True)
    i32s = J(pa.int32(), "[0, null, 0, 65536, 2147483647]")
    yield fails(i32s, pa.uint16())
    yield check(i32s, J(pa.uint16(), "[0, null, 0, 0, 65535]"), allow_int_overflow=True)


@cpp_test("IntegerUnsignedToSigned")
def _():
    u32s = J(pa.uint32(), "[4294967295, null, 0, 32768]")
    yield fails(u32s, pa.int32())
    yield fails(u32s, pa.int16())
    yield fails(u32s.slice(1), pa.int16())
    yield check(u32s, J(pa.int32(), "[-1, null, 0, 32768]"), allow_int_overflow=True)
    yield check(u32s, J(pa.int64(), "[4294967295, null, 0, 32768]"), allow_int_overflow=True)
    yield check(u32s, J(pa.int16(), "[-1, null, 0, -32768]"), allow_int_overflow=True)


@cpp_test("ToIntDowncastUnsafe")
def _():
    o = dict(allow_int_overflow=True)
    yield check(J(pa.int16(), "[0, null, 200, 1, 2]"), J(pa.uint8(), "[0, null, 200, 1, 2]"), **o)
    yield check(J(pa.int16(), "[0, null, 256, 1, 2, -1]"), J(pa.uint8(), "[0, null, 0, 1, 2, 255]"), **o)
    yield check(J(pa.int32(), "[0, null, 2000, 1, 2, -1]"), J(pa.int16(), "[0, null, 2000, 1, 2, -1]"), **o)
    yield check(J(pa.int32(), "[0, null, 2000, 70000, -70000]"), J(pa.int16(), "[0, null, 2000, 4464, -4464]"), **o)


def F(type, text):
    """ArrayFromJSON for a floating point type, including half floats."""
    values = json.loads(text)
    if type == pa.float16():
        import numpy as np

        return pa.array([None if v is None else np.float16(v) for v in values], type)
    return J(type, values)


@cpp_test("FloatingToInt")
def _():
    for from_type in [pa.float16(), pa.float32(), pa.float64()]:
        for to in [pa.int32(), pa.int64()]:
            yield check(F(from_type, "[1.0, null, 0.0, -1.0, 5.0]"), J(to, "[1, null, 0, -1, 5]"))
            yield fails(F(from_type, "[1.5, 0.0, null, 0.5, -1.5, 5.5]"), to)
            yield check(F(from_type, "[1.5, 0.0, null, 0.5, -1.5, 5.5]"), J(to, "[1, 0, null, 0, -1, 5]"), allow_float_truncate=True)


@cpp_test("FloatingToFloating")
def _():
    for from_type in [pa.float16(), pa.float32(), pa.float64()]:
        for to in [pa.float16(), pa.float32(), pa.float64()]:
            yield check(F(from_type, "[1.0, 0.0, -1.0, 5.0]"), F(to, "[1.0, 0.0, -1.0, 5.0]"))


@cpp_test("IntToFloating")
def _():
    for from_type in [pa.uint32(), pa.int32()]:
        two_24 = "[16777216, 16777217]"
        yield fails(J(from_type, two_24), pa.float32())
        yield check(J(from_type, two_24).slice(0, 1), J(pa.float32(), two_24).slice(0, 1))
    i64s = J(pa.int64(), "[-9223372036854775808, -9223372036854775807, 0, 9223372036854775806, 9223372036854775807]")
    yield fails(i64s, pa.float64())
    yield check(mask_nulls_at(i64s, [0, 1, 3, 4]), J(pa.float64(), "[null, null, 0, null, null]"))
    yield fails(J(pa.uint64(), "[9007199254740992, 9007199254740993]"), pa.float64())


@cpp_test("Decimal32ToInt")
def _():
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            yield check(J(pa.decimal32(9, 5), '["02.00000", "-11.00000", "22.00000", "-121.00000", null]'), J(pa.int32(), "[2, -11, 22, -121, null]"), **o)
    for overflow in [False, True]:
        t = J(pa.decimal32(9, 5), '["02.10000", "-11.00450", "22.00045", "-121.12100", null]')
        yield check(t, J(pa.int32(), "[2, -11, 22, -121, null]"), allow_int_overflow=overflow, allow_decimal_truncate=True)
        yield fails(t, pa.int32(), allow_int_overflow=overflow, allow_decimal_truncate=False)
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            t = J(pa.decimal32(9, 5), '["1234.00453", "9999.00344", null]')
            if truncate:
                yield check(t, J(pa.int32(), "[1234, 9999, null]"), **o)
            else:
                yield fails(t, pa.int32(), **o)
    negative_scale = decimal_unscaled(pa.decimal32(9, -3), [12345, -12000])
    yield check(negative_scale, J(pa.int32(), "[12345000, -12000000]"), allow_int_overflow=True, allow_decimal_truncate=True)


@cpp_test("Decimal64ToInt")
def _():
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            yield check(J(pa.decimal64(18, 10), '["02.0000000000", "-11.0000000000", "22.0000000000", "-121.0000000000", null]'), J(pa.int64(), "[2, -11, 22, -121, null]"), **o)
    for overflow in [False, True]:
        t = J(pa.decimal64(18, 10), '["02.1000000000", "-11.0000004500", "22.0000004500", "-121.1210000000", null]')
        yield check(t, J(pa.int32(), "[2, -11, 22, -121, null]"), allow_int_overflow=overflow, allow_decimal_truncate=True)
        yield fails(t, pa.int32(), allow_int_overflow=overflow, allow_decimal_truncate=False)
    for truncate in [False, True]:
        t = J(pa.decimal64(18, 5), '["1234567890123.00000", "9999999999999.00000", null]')
        yield check(t, J(pa.int64(), "[1234567890123, 9999999999999, null]"), allow_int_overflow=True, allow_decimal_truncate=truncate)
        yield fails(t, pa.int32(), allow_int_overflow=False, allow_decimal_truncate=truncate)
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            t = J(pa.decimal64(18, 5), '["1234567890123.45345", "9999999999999.00344", null]')
            if overflow and truncate:
                yield check(t, J(pa.int32(), "[1912276171, 1316134911, null]"), **o)
            else:
                yield fails(t, pa.int32(), **o)
    negative_scale = decimal_unscaled(pa.decimal64(18, -4), [123456789, -12])
    yield check(negative_scale, J(pa.int64(), "[1234567890000, -120000]"), allow_int_overflow=True, allow_decimal_truncate=True)


def _wide_decimal_to_int(dec_type):
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            yield check(J(dec_type(38 if dec_type is pa.decimal128 else 76, 10), '["02.0000000000", "-11.0000000000", "22.0000000000", "-121.0000000000", null]'), J(pa.int64(), "[2, -11, 22, -121, null]"), **o)
    precision = 38 if dec_type is pa.decimal128 else 76
    for overflow in [False, True]:
        t = J(dec_type(precision, 10), '["02.1000000000", "-11.0000004500", "22.0000004500", "-121.1210000000", null]')
        yield check(t, J(pa.int64(), "[2, -11, 22, -121, null]"), allow_int_overflow=overflow, allow_decimal_truncate=True)
        yield fails(t, pa.int64(), allow_int_overflow=overflow, allow_decimal_truncate=False)
    for truncate in [False, True]:
        t = J(dec_type(precision, 10), '["12345678901234567890000.0000000000", "99999999999999999999999.0000000000", null]')
        yield check(t, J(pa.int64(), "[4807115922877858896, 200376420520689663, null]"), allow_int_overflow=True, allow_decimal_truncate=truncate)
        yield fails(t, pa.int64(), allow_int_overflow=False, allow_decimal_truncate=truncate)
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            t = J(dec_type(precision, 10), '["12345678901234567890000.0045345000", "99999999999999999999999.0000344300", null]')
            if overflow and truncate:
                yield check(t, J(pa.int64(), "[4807115922877858896, 200376420520689663, null]"), **o)
            else:
                yield fails(t, pa.int64(), **o)
    negative_scale = decimal_unscaled(dec_type(precision, -4), [123456789, -12])
    yield check(negative_scale, J(pa.int64(), "[1234567890000, -120000]"), allow_int_overflow=True, allow_decimal_truncate=True)


@cpp_test("Decimal128ToInt")
def _():
    yield from _wide_decimal_to_int(pa.decimal128)


@cpp_test("Decimal256ToInt")
def _():
    t = pa.decimal256(40, 10)
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            yield check(J(t, '["02.0000000000", "-11.0000000000", "22.0000000000", "-121.0000000000", null]'), J(pa.int64(), "[2, -11, 22, -121, null]"), **o)
    for overflow in [False, True]:
        a = J(t, '["02.1000000000", "-11.0000004500", "22.0000004500", "-121.1210000000", null]')
        yield check(a, J(pa.int64(), "[2, -11, 22, -121, null]"), allow_int_overflow=overflow, allow_decimal_truncate=True)
        yield fails(a, pa.int64(), allow_int_overflow=overflow, allow_decimal_truncate=False)
    for truncate in [False, True]:
        a = J(t, '["1234567890123456789000000.0000000000", "9999999999999999999999999.0000000000", null]')
        yield check(a, J(pa.int64(), "[1096246371337547584, 1590897978359414783, null]"), allow_int_overflow=True, allow_decimal_truncate=truncate)
        yield fails(a, pa.int64(), allow_int_overflow=False, allow_decimal_truncate=truncate)
    for overflow in [False, True]:
        for truncate in [False, True]:
            o = dict(allow_int_overflow=overflow, allow_decimal_truncate=truncate)
            a = J(t, '["1234567890123456789000000.0045345000", "9999999999999999999999999.0000344300", null]')
            if overflow and truncate:
                yield check(a, J(pa.int64(), "[1096246371337547584, 1590897978359414783, null]"), **o)
            else:
                yield fails(a, pa.int64(), **o)
    negative_scale = decimal_unscaled(pa.decimal256(40, -4), [123456789, -12])
    yield check(negative_scale, J(pa.int64(), "[1234567890000, -120000]"), allow_int_overflow=True, allow_decimal_truncate=True)


@cpp_test("IntegerToDecimal")
def _():
    for decimal_type in [pa.decimal32(9, 2), pa.decimal64(18, 2), pa.decimal128(22, 2), pa.decimal256(22, 2)]:
        for integer_type in INTEGER:
            if decimal_type.bit_width > integer_type.bit_width:
                yield check(J(integer_type, "[0, 7, null, 100, 99]"), J(decimal_type, '["0.00", "7.00", null, "100.00", "99.00"]'))
    for decimal_type in [pa.decimal128(19, 0), pa.decimal256(19, 0)]:
        yield check(J(pa.int64(), [-9223372036854775808, 9223372036854775807]), J(decimal_type, '["-9223372036854775808", "9223372036854775807"]'))
    for decimal_type in [pa.decimal128(20, 0), pa.decimal256(20, 0)]:
        yield check(J(pa.uint64(), [0, 18446744073709551615]), J(decimal_type, '["0", "18446744073709551615"]'))
    yield fails(J(pa.int32(), "[0]"), pa.decimal32(9, 3))
    yield fails(J(pa.int64(), "[0]"), pa.decimal64(18, 3))
    yield fails(J(pa.int8(), "[0]"), pa.decimal128(5, 3))
    yield fails(J(pa.int32(), "[0]"), pa.decimal256(76, 67))


def _fraction(scale, whole):
    return f"{whole}.{'1234567890'[:scale]}"


def _decimal_to_decimal(no_truncation, small, truncation, whole, invalid, both_directions):
    """The shared body of the C++ `DecimalNToDecimalM` tests, given each
    test's types: `(from, to)` pairs for the exact conversions, the
    truncating pair and its whole-number round trip, and the `(from, to
    constructor)` of the conversions that overflow the target precision."""
    for truncate in [False, True]:
        a = J(no_truncation[0], "[2, 30, 22, -121, null]")
        b = J(no_truncation[1], "[2, 30, 22, -121, null]")
        yield check(a, b, allow_decimal_truncate=truncate)
        if both_directions:
            yield check(b, a, allow_decimal_truncate=truncate)
    for truncate in [False, True]:
        for from_type, to in small:
            a = J(from_type, '["12.34", "0.56"]')
            b = J(to, '["12.34", "0.56"]')
            yield check(a, b, allow_decimal_truncate=truncate)
            if both_directions:
                yield check(b, a, allow_decimal_truncate=truncate)
    from_type, to = truncation
    fractional = J(from_type, [_fraction(from_type.scale, "-02"), _fraction(from_type.scale, "30"), None])
    truncated = J(to, '["-2", "30", null]')
    whole_from, whole_to = whole
    yield check(fractional, truncated, allow_decimal_truncate=True)
    yield check(J(whole_from, '["-2", "30", null]'), J(whole_to, '["-2", "30", null]'), allow_decimal_truncate=True)
    yield fails(fractional, to, allow_decimal_truncate=False)
    yield check(J(whole_from, '["-2", "30", null]'), J(whole_to, '["-2", "30", null]'), allow_decimal_truncate=False)
    invalid_from, invalid_to = invalid
    a = J(invalid_from(4, 2), '["12.34"]')
    for precision, scale in [(3, 2), (4, 3), (2, 1)]:
        # C++ expects a result that fails validation; it is compared as is.
        yield attempt(a, invalid_to(precision, scale), "overflows precision", allow_decimal_truncate=True)
        yield fails(a, invalid_to(precision, scale), allow_decimal_truncate=False)


d32, d64, d128, d256 = pa.decimal32, pa.decimal64, pa.decimal128, pa.decimal256
for _name, _arguments in {
    "Decimal32ToDecimal32": ((d32(9, 5), d32(9, 0)), [(d32(5, 2), d32(4, 2))], (d32(9, 5), d32(6, 0)), (d32(6, 0), d32(9, 5)), (d32, d32), True),
    "Decimal64ToDecimal64": ((d64(18, 10), d64(9, 0)), [(d64(5, 2), d64(4, 2))], (d64(18, 10), d64(12, 0)), (d64(12, 0), d64(18, 10)), (d64, d64), True),
    "Decimal128ToDecimal128": ((d128(38, 10), d128(28, 0)), [(d128(5, 2), d128(4, 2))], (d128(38, 10), d128(28, 0)), (d128(28, 0), d128(38, 10)), (d128, d128), True),
    "Decimal256ToDecimal256": ((d256(38, 10), d256(28, 0)), [(d256(5, 2), d256(4, 2))], (d256(38, 10), d256(28, 0)), (d256(28, 0), d256(38, 10)), (d256, d256), True),
    "Decimal32ToDecimal64": ((d32(9, 5), d64(16, 0)), [(d32(5, 2), d64(4, 2)), (d32(5, 2), d64(16, 2))], (d32(7, 5), d64(14, 0)), (d32(9, 0), d64(18, 10)), (d32, d64), False),
    "Decimal32ToDecimal128": ((d32(9, 5), d128(16, 0)), [(d32(5, 2), d128(4, 2)), (d32(5, 2), d128(16, 2))], (d32(7, 5), d128(14, 0)), (d32(9, 0), d128(38, 10)), (d32, d128), False),
    "Decimal32ToDecimal256": ((d32(9, 5), d256(16, 0)), [(d32(5, 2), d256(4, 2)), (d32(5, 2), d256(16, 2))], (d32(7, 5), d256(14, 0)), (d32(9, 0), d256(76, 10)), (d32, d256), False),
    "Decimal64ToDecimal32": ((d64(18, 5), d32(9, 0)), [(d64(12, 2), d32(4, 2))], (d64(15, 5), d32(6, 0)), (d64(12, 0), d32(9, 5)), (d64, d32), False),
    "Decimal64ToDecimal128": ((d64(18, 10), d128(28, 0)), [(d64(5, 2), d128(4, 2)), (d64(5, 2), d128(16, 2))], (d64(16, 10), d128(14, 0)), (d64(18, 0), d128(38, 10)), (d64, d128), False),
    "Decimal64ToDecimal256": ((d64(18, 10), d256(16, 0)), [(d64(5, 2), d256(4, 2)), (d64(5, 2), d256(16, 2))], (d64(16, 10), d256(14, 0)), (d64(18, 0), d256(76, 10)), (d64, d256), False),
    "Decimal128ToDecimal32": ((d128(26, 5), d32(9, 0)), [(d128(28, 2), d32(4, 2))], (d128(28, 5), d32(7, 0)), (d128(22, 0), d32(9, 5)), (d128, d32), False),
    "Decimal128ToDecimal64": ((d128(26, 10), d64(15, 0)), [(d128(28, 2), d64(4, 2))], (d128(28, 10), d64(12, 0)), (d128(22, 0), d64(18, 10)), (d128, d64), False),
    "Decimal128ToDecimal256": ((d128(38, 10), d256(48, 0)), [(d128(5, 2), d256(4, 2)), (d128(5, 2), d256(40, 2))], (d128(38, 10), d256(28, 0)), (d128(28, 0), d256(38, 10)), (d128, d256), False),
    "Decimal256ToDecimal32": ((d256(42, 5), d32(9, 0)), [(d256(42, 2), d32(4, 2))], (d256(52, 5), d32(7, 0)), (d256(42, 0), d32(9, 5)), (d256, d32), False),
    "Decimal256ToDecimal64": ((d256(42, 10), d64(15, 0)), [(d256(42, 2), d64(4, 2))], (d256(52, 10), d64(12, 0)), (d256(42, 0), d64(18, 10)), (d256, d64), False),
    "Decimal256ToDecimal128": ((d256(42, 10), d128(28, 0)), [(d256(42, 2), d128(4, 2))], (d256(52, 10), d128(28, 0)), (d256(42, 0), d128(38, 10)), (d256, d128), False),
}.items():
    cpp_test(_name)(lambda _arguments=_arguments: _decimal_to_decimal(*_arguments))


DECIMALS_5_2 = [pa.decimal32(5, 2), pa.decimal64(5, 2), pa.decimal128(5, 2), pa.decimal256(5, 2)]


@cpp_test("FloatingToDecimal")
def _():
    for float_type in [pa.float32(), pa.float64()]:
        for decimal_type in DECIMALS_5_2:
            yield check(J(float_type, "[0.0, null, 123.45, 123.456, 999.994]"), J(decimal_type, '["0.00", null, "123.45", "123.46", "999.99"]'))
            yield fails(J(float_type, "[999.996]"), decimal_type)
            yield check(J(float_type, "[0.0, null, 999.996, 123.45, 999.994]"), J(decimal_type, '["0.00", null, "0.00", "123.45", "999.99"]'), allow_decimal_truncate=True)
    for decimal_type in [pa.decimal128, pa.decimal256]:
        yield check(J(pa.float32(), "[1.8446746e+19, -1.8446746e+19]"), J(decimal_type(20, 0), '["18446746272732807168", "-18446746272732807168"]'))
        yield check(J(pa.float64(), "[1.8446744073709556e+19, -1.8446744073709556e+19]"), J(decimal_type(20, 0), '["18446744073709555712", "-18446744073709555712"]'))
        yield check(J(pa.float32(), "[1.8446746e+15, -1.8446746e+15]"), J(decimal_type(20, 4), '["1844674629206016.0000", "-1844674629206016.0000"]'))
        yield check(J(pa.float64(), "[1.8446744073709556e+15, -1.8446744073709556e+15]"), J(decimal_type(20, 4), '["1844674407370955.5000", "-1844674407370955.5000"]'))


@cpp_test("DecimalToFloating")
def _():
    for float_type in [pa.float32(), pa.float64()]:
        for decimal_type in DECIMALS_5_2:
            yield check(J(decimal_type, '["0.00", null, "123.45", "999.99"]'), J(float_type, "[0.0, null, 123.45, 999.99]"))


@cpp_test("DecimalToString")
def _():
    for string_type in [pa.utf8(), pa.string_view(), pa.large_utf8()]:
        for decimal_type in DECIMALS_5_2:
            yield check(J(decimal_type, '["0.00", null, "123.45", "999.99"]'), J(string_type, '["0.00", null, "123.45", "999.99"]'))


# One type of every Arrow C++ type id PyArrow can construct, in C++ id order.
REGISTRATION_TYPES = [
    pa.null(), pa.bool_(), *INTEGER, pa.float16(), pa.float32(), pa.float64(),
    pa.decimal32(5, 2), pa.decimal64(5, 2), pa.decimal128(5, 2), pa.decimal256(5, 2),
    pa.utf8(), pa.large_utf8(), pa.string_view(), pa.binary(), pa.large_binary(), pa.binary_view(),
    pa.binary(3), pa.date32(), pa.date64(), pa.time32("s"), pa.time64("us"), pa.timestamp("s"),
    pa.duration("s"), pa.month_day_nano_interval(), pa.list_(pa.int8()), pa.large_list(pa.int8()),
    pa.list_view(pa.int8()), pa.large_list_view(pa.int8()), pa.list_(pa.int8(), 2),
    pa.map_(pa.utf8(), pa.int8()), pa.struct([("a", pa.int8())]),
    pa.sparse_union([pa.field("a", pa.int8())]), pa.dense_union([pa.field("a", pa.int8())]),
    pa.dictionary(pa.int8(), pa.utf8()), pa.run_end_encoded(pa.int32(), pa.int8()),
]


def registration_sweep():
    """Beyond the C++ tests: the cast of an all-null array between every pair
    of `REGISTRATION_TYPES`, so the Rust support table is compared with the
    C++ cast function table for every pair of type ids."""
    for from_type in REGISTRATION_TYPES:
        for to in REGISTRATION_TYPES:
            if from_type != to:
                yield Case(pa.nulls(2, from_type), to, {}, None, f"{from_type} to {to}", variants=False)


_REGISTRY.append(("RegistrationTable", registration_sweep))


def cases():
    """Every `(C++ test, case number, Case)` of the corpus."""
    for name, function in _REGISTRY:
        for number, case in enumerate(function()):
            yield name, number, case


def unmapped_tests():
    translated = {name for name, _ in _REGISTRY}
    return [name for name in CPP_TESTS if name not in translated and name not in MAPPED_ELSEWHERE]
