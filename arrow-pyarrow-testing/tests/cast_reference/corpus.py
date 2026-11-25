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

Translated from `cpp/src/arrow/compute/kernels/scalar_cast_test.cc` and the
cast tests of `cpp/src/arrow/scalar_test.cc` of Apache Arrow 24.0.0 (tag `apache-arrow-24.0.0`, commit
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

import numpy as np
import pyarrow as pa

# Every cast TEST of `cpp/src/arrow/scalar_test.cc` at the pinned commit, in
# source order: scalar casts through the same cast function.
SCALAR_CPP_TESTS = """
TestNullScalar.Cast TestBooleanScalar.Cast TestScalar.IdentityCast
TestDecimalScalar.Cast TestBinaryScalar.Cast TestFixedSizeBinaryScalar.Cast
TestDateScalars.MakeScalar TestTimeScalars.MakeScalar TestTimestampScalars.Cast
TestNumericScalar.Cast TestListLikeScalar.Cast TestFixedSizeListScalar.Cast
TestMapScalar.Cast TestStructScalar.Cast TestDictionaryScalar.Cast
TestUnionScalar.Cast
""".split()

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

SCALAR_TO_STRING = (
    "no Rust equivalent: the test asserts only C++ Scalar::CastTo to utf8, "
    "which formats Scalar::ToString outside the cast function, as the test "
    "itself notes"
)

# C++ tests whose assertions are not cast outcomes, and the Rust test owning
# the equivalent assertion.
MAPPED_ELSEWHERE = {
    "SameTypeZeroCopy": "arrow-cast cast::tests::conversion_casts_reuse_buffers",
    "NoOutBitmapIfInIsAllValid": "arrow-cast cast::tests::conversion_casts_reuse_buffers",
    "TestStructScalar.Cast": SCALAR_TO_STRING,
    "TestUnionScalar.Cast": SCALAR_TO_STRING,
    "CastDoesNotProvideDefaultOptions": (
        "the signature of cast_with_conversion_options, which takes "
        "&ConversionOptions, so no call can omit them"
    ),
}

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
        self.exception = None


def check(input, expected, label="", **options):
    return Case(input, expected.type, options, expected, label or "check")


def fails(input, to, label="", **options):
    return Case(input, to, options, "Invalid", label or "fails")


def unsupported(input, to, label=""):
    return Case(input, to, {}, "NotImplemented", label or "unsupported")


def type_error(input, to, label="", **options):
    return Case(input, to, options, "TypeError", label or "type error")


def attempt(input, to, label="", **options):
    return Case(input, to, options, None, label or "attempt")


# Why an approved exception lets Rust refuse as `Invalid` where Arrow C++ converts.
INVALID_UTF8_STRINGS = (
    "approved exception: Arrow C++ converts to strings that fail ValidateFull, "
    "which a Rust string array cannot soundly hold, so Rust refuses them "
    "whatever allow_invalid_utf8 is"
)


def excepted(input, to, label, reason, **options):
    """A C++ cast for which an approved exception lets Rust refuse as
    `Invalid` where Arrow C++ converts, for `reason`. Any other difference is
    still a disagreement."""
    case = Case(input, to, options, None, label)
    case.exception = reason
    return case


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
        # C++ reads timestamp strings as UTC instants whatever the zone.
        return pa.array(values, pa.string()).cast(pa.timestamp(type.unit)).view(type)
    if pa.types.is_dictionary(type):
        return pa.array([_value(type.value_type, v) for v in values], type.value_type).dictionary_encode().cast(type)
    return pa.array([_value(type, v) for v in values], type)


def D(type, indices, dictionary):
    """C++ `DictArrayFromJSON(type, indices, dictionary)`."""
    return pa.DictionaryArray.from_arrays(J(type.index_type, indices), J(type.value_type, dictionary))


def mask_nulls_at(array, indices):
    """C++ `MaskArrayWithNullsAt`, and `TweakValidityBit(array, index, false)`
    for one index: `array` with `indices` made null and its own buffers and
    children unchanged."""
    assert array.offset == 0
    valid = [array[i].is_valid and i not in indices for i in range(len(array))]
    bitmap = pa.array(valid, pa.bool_()).buffers()[1]
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        own, children = array.buffers()[1:2], [array.values]
    elif pa.types.is_fixed_size_list(array.type):
        own, children = [], [array.values]
    else:
        own, children = array.buffers()[1:], None
    return pa.Array.from_buffers(array.type, len(array), [bitmap] + own, children=children)


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


def _coarse_and_fine(make, pairs):
    """The C++ unit promotion and truncation loops shared by the timestamp,
    time and duration tests: `pairs` of `(coarse, fine, promoted, truncated)`."""
    for coarse, fine, promoted, truncated in pairs:
        coarse_values = J(coarse, "[0, null, 200, 1, 2]")
        yield check(coarse_values, J(fine, promoted))
        yield fails(J(fine, truncated), coarse)
        yield check(J(fine, truncated), coarse_values, allow_time_truncate=True)


@cpp_test("TimestampToTimestamp")
def _():
    ts = pa.timestamp
    yield from _coarse_and_fine(ts, [
        (ts("s"), ts("ms"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (ts("ms"), ts("us"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (ts("us"), ts("ns"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
    ])
    yield check(J(ts("s", "UTC"), "[0, null, 200, 1, 2]"), J(ts("s"), "[0, null, 200, 1, 2]"))
    yield from _coarse_and_fine(ts, [
        (ts("s"), ts("us"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200456000, 1123000, 2456000]"),
        (ts("ms"), ts("ns"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200456000, 1123000, 2456000]"),
    ])
    yield from _coarse_and_fine(ts, [
        (ts("s"), ts("ns"), "[0, null, 200000000000, 1000000000, 2000000000]", "[0, null, 200456000000, 1123000000, 2456000000]"),
    ])


@cpp_test("TimestampZeroCopy")
def _():
    # The zero copy assertion is `cast::tests::conversion_casts_reuse_buffers`.
    values = "[0, null, 2000, 1000, 0]"
    yield check(J(pa.timestamp("s"), values), J(pa.timestamp("s"), values))
    yield check(J(pa.timestamp("s"), values), J(pa.int64(), values))
    yield check(J(pa.int64(), values), J(pa.timestamp("s"), values))
    yield check(J(pa.timestamp("s", "UTC"), values), J(pa.timestamp("s"), values))


@cpp_test("TimestampToTimestampMultiplyOverflow")
def _():
    yield fails(J(pa.timestamp("s"), "[-30610224000, -5364662400, 946684800, 10413792000, 32503680000]"), pa.timestamp("ns"))


TIMESTAMP_JSON = [
    "1970-01-01T00:00:59.123456789", "2000-02-29T23:23:23.999999999",
    "1899-01-01T00:59:20.001001001", "2033-05-18T03:33:20.000000000",
    "2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
    "2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004132",
    "2010-01-01T05:25:25.005321", "2010-01-03T06:30:30.006163",
    "2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
    "2008-12-28", "2008-12-29", "2012-01-01 01:02:03", None,
]
TIMESTAMP_SECONDS_JSON = [
    "1970-01-01T00:00:59", "2000-02-29T23:23:23",
    "1899-01-01T00:59:20", "2033-05-18T03:33:20",
    "2020-01-01T01:05:05", "2019-12-31T02:10:10",
    "2019-12-30T03:15:15", "2009-12-31T04:20:20",
    "2010-01-01T05:25:25", "2010-01-03T06:30:30",
    "2010-01-04T07:35:35", "2006-01-01T08:40:40",
    "2005-12-31T09:45:45", "2008-12-28", "2008-12-29",
    "2012-01-01 01:02:03", None,
]
TIMESTAMP_EXTREME_JSON = ["1677-09-20T00:00:59.123456", "2262-04-13T23:23:23.999999"]
UNITS = ["s", "ms", "us", "ns"]


@cpp_test("TimestampToDate")
def _():
    timestamps = J(pa.timestamp("ns"), TIMESTAMP_JSON)
    date_32 = J(pa.date32(), "[0, 11016, -25932, 23148, 18262, 18261, 18260, 14609, 14610, 14612, 14613, 13149, 13148, 14241, 14242, 15340, null]")
    date_64 = J(pa.date64(), "[0, 951782400000, -2240524800000, 1999987200000, 1577836800000, 1577750400000, 1577664000000, 1262217600000, 1262304000000, 1262476800000, 1262563200000, 1136073600000, 1135987200000, 1230422400000, 1230508800000, 1325376000000, null]")
    extreme = J(pa.timestamp("us"), TIMESTAMP_EXTREME_JSON)
    yield check(timestamps, date_32)
    yield check(timestamps, date_64)
    yield check(extreme, J(pa.date32(), "[-106753, 106753]"))
    yield check(extreme, J(pa.date64(), "[-9223459200000, 9223459200000]"))
    for unit in UNITS:
        yield check(J(pa.timestamp(unit), TIMESTAMP_SECONDS_JSON), date_32)
        yield check(J(pa.timestamp(unit), TIMESTAMP_SECONDS_JSON), date_64)


@cpp_test("ZonedTimestampToDate")
def _():
    timestamps = J(pa.timestamp("ns", "Pacific/Marquesas"), TIMESTAMP_JSON)
    yield check(timestamps, J(pa.date32(), "[-1, 11016, -25933, 23147, 18261, 18260, 18259, 14608, 14609, 14611, 14612, 13148, 13148, 14240, 14241, 15339, null]"))
    yield check(timestamps, J(pa.date64(), "[-86400000, 951782400000, -2240611200000, 1999900800000, 1577750400000, 1577664000000, 1577577600000, 1262131200000, 1262217600000, 1262390400000, 1262476800000, 1135987200000, 1135987200000, 1230336000000, 1230422400000, 1325289600000, null]"))
    date_32 = J(pa.date32(), "[0, 11017, -25932, 23148, 18262, 18261, 18260, 14609, 14610, 14612, 14613, 13149, 13148, 14241, 14242, 15340, null]")
    date_64 = J(pa.date64(), "[0, 951868800000, -2240524800000, 1999987200000, 1577836800000, 1577750400000, 1577664000000, 1262217600000, 1262304000000, 1262476800000, 1262563200000, 1136073600000, 1135987200000, 1230422400000, 1230508800000, 1325376000000, null]")
    for unit in UNITS:
        timestamps = J(pa.timestamp(unit, "Australia/Broken_Hill"), TIMESTAMP_SECONDS_JSON)
        yield check(timestamps, date_32)
        yield check(timestamps, date_64)
    for unit in UNITS:
        timestamps = J(pa.timestamp(unit, "Mars/Mariner_Valley"), TIMESTAMP_SECONDS_JSON)
        yield fails(timestamps, pa.date32(), **UNSAFE)
        yield fails(timestamps, pa.date64(), **UNSAFE)


@cpp_test("TimestampToTime")
def _():
    t32, t64, ts = pa.time32, pa.time64, pa.timestamp
    timestamps = J(ts("ns"), TIMESTAMP_JSON)
    extreme = J(ts("us"), TIMESTAMP_EXTREME_JSON)
    timestamps_us = J(ts("us"), [
        "1970-01-01T00:00:59.123456", "2000-02-29T23:23:23.999999",
        "1899-01-01T00:59:20.001001", "2033-05-18T03:33:20.000000",
        "2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
        "2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004132",
        "2010-01-01T05:25:25.005321", "2010-01-03T06:30:30.006163",
        "2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
        "2008-12-28", "2008-12-29", "2012-01-01 01:02:03", None])
    timestamps_ms = J(ts("ms"), [
        "1970-01-01T00:00:59.123", "2000-02-29T23:23:23.999",
        "1899-01-01T00:59:20.001", "2033-05-18T03:33:20.000",
        "2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
        "2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004",
        "2010-01-01T05:25:25.005", "2010-01-03T06:30:30.006",
        "2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
        "2008-12-28", "2008-12-29", "2012-01-01 01:02:03", None])
    timestamps_s = J(ts("s"), TIMESTAMP_SECONDS_JSON)
    times = J(t64("ns"), "[59123456789, 84203999999999, 3560001001001, 12800000000000, 3905001000000, 7810002000000, 11715003000000, 15620004132000, 19525005321000, 23430006163000, 27335000000000, 31240000000000, 35145000000000, 0, 0, 3723000000000, null]")
    times_ns_us = J(t64("us"), "[59123456, 84203999999, 3560001001, 12800000000, 3905001000, 7810002000, 11715003000, 15620004132, 19525005321, 23430006163, 27335000000, 31240000000, 35145000000, 0, 0, 3723000000, null]")
    times_ns_ms = J(t32("ms"), "[59123, 84203999, 3560001, 12800000, 3905001, 7810002, 11715003, 15620004, 19525005, 23430006, 27335000, 31240000, 35145000, 0, 0, 3723000, null]")
    times_us_ns = J(t64("ns"), "[59123456000, 84203999999000, 3560001001000, 12800000000000, 3905001000000, 7810002000000, 11715003000000, 15620004132000, 19525005321000, 23430006163000, 27335000000000, 31240000000000, 35145000000000, 0, 0, 3723000000000, null]")
    times_ms_ns = J(t64("ns"), "[59123000000, 84203999000000, 3560001000000, 12800000000000, 3905001000000, 7810002000000, 11715003000000, 15620004000000, 19525005000000, 23430006000000, 27335000000000, 31240000000000, 35145000000000, 0, 0, 3723000000000, null]")
    times_ms_us = J(t64("us"), "[59123000, 84203999000, 3560001000, 12800000000, 3905001000, 7810002000, 11715003000, 15620004000, 19525005000, 23430006000, 27335000000, 31240000000, 35145000000, 0, 0, 3723000000, null]")
    times_extreme = J(t64("us"), "[59123456, 84203999999]")
    times_s = J(t32("s"), "[59, 84203, 3560, 12800, 3905, 7810, 11715, 15620, 19525, 23430, 27335, 31240, 35145, 0, 0, 3723, null]")
    times_ms = J(t32("ms"), "[59000, 84203000, 3560000, 12800000, 3905000, 7810000, 11715000, 15620000, 19525000, 23430000, 27335000, 31240000, 35145000, 0, 0, 3723000, null]")
    times_us = J(t64("us"), "[59000000, 84203000000, 3560000000, 12800000000, 3905000000, 7810000000, 11715000000, 15620000000, 19525000000, 23430000000, 27335000000, 31240000000, 35145000000, 0, 0, 3723000000, null]")
    times_ns = J(t64("ns"), "[59000000000, 84203000000000, 3560000000000, 12800000000000, 3905000000000, 7810000000000, 11715000000000, 15620000000000, 19525000000000, 23430000000000, 27335000000000, 31240000000000, 35145000000000, 0, 0, 3723000000000, null]")

    yield check(timestamps, times)
    yield fails(timestamps, t64("us"))
    yield check(extreme, times_extreme)
    yield check(J(ts("s"), TIMESTAMP_SECONDS_JSON), times_s)
    yield check(J(ts("s"), TIMESTAMP_SECONDS_JSON), times_ms)
    yield check(J(ts("ms"), TIMESTAMP_SECONDS_JSON), times_s)
    yield check(J(ts("ms"), TIMESTAMP_SECONDS_JSON), times_ms)
    for unit, expected in [("us", times_us), ("us", times_ns), ("us", times_ms), ("us", times_s), ("ns", times_ns), ("ns", times_us), ("ns", times_ms), ("ns", times_s)]:
        yield check(J(ts(unit), TIMESTAMP_SECONDS_JSON), expected)

    yield fails(timestamps, t64("us"))
    yield fails(timestamps, t32("ms"))
    yield fails(timestamps, t32("s"))
    yield fails(timestamps_us, t32("ms"))
    yield fails(timestamps_us, t32("s"))
    yield fails(timestamps_ms, t32("s"))
    yield check(timestamps, times_ns_us, allow_time_truncate=True)
    yield check(timestamps, times_ns_ms, allow_time_truncate=True)
    yield check(timestamps, times_s, allow_time_truncate=True)
    yield check(timestamps_us, times_ns_ms, allow_time_truncate=True)
    yield check(timestamps_us, times_s, allow_time_truncate=True)
    yield check(timestamps_ms, times_s, allow_time_truncate=True)

    yield check(timestamps_us, times_us_ns)
    yield check(timestamps_ms, times_ms_ns)
    yield check(timestamps_ms, times_ms_us)
    yield check(timestamps_s, times_ns)
    yield check(timestamps_s, times_us)
    yield check(timestamps_s, times_ms)

    for unit in UNITS:
        timestamps = J(ts(unit, "Mars/Mariner_Valley"), TIMESTAMP_SECONDS_JSON)
        yield fails(timestamps, t32(unit) if unit in ("s", "ms") else t64(unit), **UNSAFE)


@cpp_test("ZonedTimestampToTime")
def _():
    yield check(
        J(pa.timestamp("ns", "Pacific/Marquesas"), TIMESTAMP_JSON),
        J(pa.time64("ns"), "[52259123456789, 50003999999999, 56480001001001, 65000000000000, 56105001000000, 60010002000000, 63915003000000, 67820004132000, 71725005321000, 75630006163000, 79535000000000, 83440000000000, 945000000000, 52200000000000, 52200000000000, 55923000000000, null]"),
    )
    seconds = [34259, 35603, 35960, 47000, 41705, 45610, 49515, 53420, 57325, 61230, 65135, 69040, 72945, 37800, 37800, 41523]
    for unit, type, factor in [("s", pa.time32("s"), 1), ("ms", pa.time32("ms"), 1000), ("us", pa.time64("us"), 10**6), ("ns", pa.time64("ns"), 10**9)]:
        yield check(
            J(pa.timestamp(unit, "Australia/Broken_Hill"), TIMESTAMP_SECONDS_JSON),
            J(type, [v * factor for v in seconds] + [None]),
        )


@cpp_test("TimeToTime")
def _():
    t32, t64 = pa.time32, pa.time64
    yield from _coarse_and_fine(None, [
        (t32("s"), t32("ms"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (t32("ms"), t64("us"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (t64("us"), t64("ns"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (t32("s"), t64("us"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200456000, 1123000, 2456000]"),
        (t32("ms"), t64("ns"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200456000, 1123000, 2456000]"),
        (t32("s"), t64("ns"), "[0, null, 200000000000, 1000000000, 2000000000]", "[0, null, 200456000000, 1123000000, 2456000000]"),
    ])


@cpp_test("TimeZeroCopy")
def _():
    # The zero copy assertion is `cast::tests::conversion_casts_reuse_buffers`.
    values = "[0, null, 2000, 1000, 0]"
    for source, target in [(pa.time32("s"), pa.time32("s")), (pa.time32("s"), pa.int32()), (pa.int32(), pa.time32("s")),
                           (pa.time64("us"), pa.time64("us")), (pa.time64("us"), pa.int64()), (pa.int64(), pa.time64("us"))]:
        yield check(J(source, values), J(target, values))


STRING_TYPES = [pa.utf8(), pa.string_view(), pa.large_utf8()]


@cpp_test("DateToString")
def _():
    for string_type in STRING_TYPES:
        yield check(J(pa.date32(), "[0, null]"), J(string_type, '["1970-01-01", null]'))
        yield check(J(pa.date64(), "[86400000, null]"), J(string_type, '["1970-01-02", null]'))


@cpp_test("TimeToString")
def _():
    for string_type in STRING_TYPES:
        yield check(J(pa.time32("s"), "[1, 62]"), J(string_type, '["00:00:01", "00:01:02"]'))
        yield check(J(pa.time64("ns"), "[0, 1]"), J(string_type, '["00:00:00.000000000", "00:00:00.000000001"]'))


@cpp_test("TimestampToString")
def _():
    for string_type in STRING_TYPES:
        yield check(J(pa.timestamp("s"), "[-30610224000, -5364662400]"), J(string_type, '["1000-01-01 00:00:00", "1800-01-01 00:00:00"]'))
        yield check(J(pa.timestamp("ms"), "[-30610224000000, -5364662400000]"), J(string_type, '["1000-01-01 00:00:00.000", "1800-01-01 00:00:00.000"]'))
        yield check(J(pa.timestamp("us"), "[-30610224000000000, -5364662400000000]"), J(string_type, '["1000-01-01 00:00:00.000000", "1800-01-01 00:00:00.000000"]'))
        yield check(J(pa.timestamp("ns"), "[-596933876543210988, 349837323456789012]"), J(string_type, '["1951-02-01 01:02:03.456789012", "1981-02-01 01:02:03.456789012"]'))


@cpp_test("TimestampWithZoneToString")
def _():
    for string_type in STRING_TYPES:
        yield check(J(pa.timestamp("s", "UTC"), "[-30610224000, -5364662400]"), J(string_type, '["1000-01-01 00:00:00Z", "1800-01-01 00:00:00Z"]'))
        yield check(J(pa.timestamp("s", "America/Phoenix"), "[-34226955, 1456767743]"), J(string_type, '["1968-11-30 13:30:45-0700", "2016-02-29 10:42:23-0700"]'))
        yield check(J(pa.timestamp("ms", "America/Phoenix"), "[-34226955877, 1456767743456]"), J(string_type, '["1968-11-30 13:30:44.123-0700", "2016-02-29 10:42:23.456-0700"]'))
        yield check(J(pa.timestamp("us", "America/Phoenix"), "[-34226955877000, 1456767743456789]"), J(string_type, '["1968-11-30 13:30:44.123000-0700", "2016-02-29 10:42:23.456789-0700"]'))
        yield check(J(pa.timestamp("ns", "America/Phoenix"), "[-34226955876543211, 1456767743456789246]"), J(string_type, '["1968-11-30 13:30:44.123456789-0700", "2016-02-29 10:42:23.456789246-0700"]'))


@cpp_test("DateToDate")
def _():
    day_32 = J(pa.date32(), "[0, null, 100, 1, 10]")
    day_64 = J(pa.date64(), "[0, null, 8640000000, 86400000, 864000000]")
    yield check(day_32, day_32)
    yield check(day_64, day_64)
    yield check(day_32, day_64)
    yield check(day_64, day_32)
    truncated = J(pa.date64(), "[0, null, 8640000123, 86400456, 864000789]")
    yield fails(truncated, pa.date32())
    yield check(truncated, day_32, allow_time_truncate=True)


@cpp_test("DateZeroCopy")
def _():
    # The zero copy assertion is `cast::tests::conversion_casts_reuse_buffers`.
    for source, target, values in [
        (pa.date32(), pa.date32(), "[0, null, 2000, 1000, 0]"), (pa.date32(), pa.int32(), "[0, null, 2000, 1000, 0]"),
        (pa.int32(), pa.date32(), "[0, null, 2000, 1000, 0]"),
        (pa.date64(), pa.date64(), "[0, null, 172800000, 86400000, 0]"), (pa.date64(), pa.int64(), "[0, null, 172800000, 86400000, 0]"),
        (pa.int64(), pa.date64(), "[0, null, 172800000, 86400000, 0]"),
    ]:
        yield check(J(source, values), J(target, values))


@cpp_test("DurationToDuration")
def _():
    d = pa.duration
    yield from _coarse_and_fine(None, [
        (d("s"), d("ms"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (d("ms"), d("us"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (d("us"), d("ns"), "[0, null, 200000, 1000, 2000]", "[0, null, 200456, 1123, 2456]"),
        (d("s"), d("us"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200000456, 1000123, 2000456]"),
        (d("ms"), d("ns"), "[0, null, 200000000, 1000000, 2000000]", "[0, null, 200000456, 1000123, 2000456]"),
        (d("s"), d("ns"), "[0, null, 200000000000, 1000000000, 2000000000]", "[0, null, 200000000456, 1000000123, 2000000456]"),
    ])


@cpp_test("DurationZeroCopy")
def _():
    # The zero copy assertion is `cast::tests::conversion_casts_reuse_buffers`.
    values = "[0, null, 2000, 1000, 0]"
    for source, target in [(pa.duration("s"), pa.duration("s")), (pa.duration("s"), pa.int64()), (pa.int64(), pa.duration("s"))]:
        yield check(J(source, values), J(target, values))


@cpp_test("DurationToDurationMultiplyOverflow")
def _():
    yield fails(J(pa.duration("s"), "[10000000000, 1, 2, 3, 10000000000]"), pa.duration("ns"))


@cpp_test("DurationToString")
def _():
    for string_type in STRING_TYPES:
        for unit in UNITS:
            yield check(J(pa.duration(unit), "[0, null, 1234567, 2000]"), J(string_type, '["0", null, "1234567", "2000"]'))


@cpp_test("MiscToFloating")
def _():
    for to in [pa.float32(), pa.float64()]:
        yield check(J(pa.int16(), "[0, null, 200, 1, 2]"), J(to, "[0, null, 200, 1, 2]"))
        yield check(J(pa.float32(), "[0, null, 200, 1, 2]"), J(to, "[0, null, 200, 1, 2]"))
        yield check(J(pa.bool_(), "[true, null, false, false, true]"), J(to, "[1, null, 0, 0, 1]"))


@cpp_test("UnsupportedInputType")
def _():
    yield unsupported(J(pa.int32(), "[1, 2, 3]"), pa.list_(pa.utf8()))


@cpp_test("UnsupportedTargetType")
def _():
    yield unsupported(J(pa.int32(), "[1, 2, 3]"), pa.dense_union([pa.field("a", pa.int32())]))


@cpp_test("StringToBoolean")
def _():
    for string_type in [pa.utf8(), pa.large_utf8()]:
        yield check(J(string_type, '["False", null, "true", "True", "false"]'), J(pa.bool_(), "[false, null, true, true, false]"))
        yield check(J(string_type, '["0", null, "1", "1", "0"]'), J(pa.bool_(), "[false, null, true, true, false]"))
        yield fails(J(string_type, '["false "]'), pa.bool_())
        yield fails(J(string_type, '["T"]'), pa.bool_())


@cpp_test("StringToInt")
def _():
    for string_type in [pa.utf8(), pa.large_utf8()]:
        for signed_type in [pa.int8(), pa.int16(), pa.int32(), pa.int64()]:
            yield check(J(string_type, '["0", null, "127", "-1", "0", "0x0", "0x7F"]'), J(signed_type, "[0, null, 127, -1, 0, 0, 127]"))
        yield check(
            J(string_type, '["2147483647", null, "-2147483648", "0", "0X0", "0x7FFFFFFF", "0XFFFFfFfF", "0Xf0000000"]'),
            J(pa.int32(), "[2147483647, null, -2147483648, 0, 0, 2147483647, -1, -268435456]"),
        )
        yield check(
            J(string_type, '["9223372036854775807", null, "-9223372036854775808", "0", "0x0", "0x7FFFFFFFFFFFFFFf", "0XF000000000000001"]'),
            J(pa.int64(), "[9223372036854775807, null, -9223372036854775808, 0, 0, 9223372036854775807, -1152921504606846975]"),
        )
        for unsigned_type in [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]:
            yield check(J(string_type, '["0", null, "127", "255", "0", "0X0", "0xff", "0x7f"]'), J(unsigned_type, "[0, null, 127, 255, 0, 0, 255, 127]"))
        yield check(
            J(string_type, '["2147483647", null, "4294967295", "0", "0x0", "0x7FFFFFFf", "0xFFFFFFFF"]'),
            J(pa.uint32(), "[2147483647, null, 4294967295, 0, 0, 2147483647, 4294967295]"),
        )
        yield check(
            J(string_type, '["9223372036854775807", null, "18446744073709551615", "0", "0x0", "0x7FFFFFFFFFFFFFFf", "0xfFFFFFFFFFFFFFFf"]'),
            J(pa.uint64(), "[9223372036854775807, null, 18446744073709551615, 0, 0, 9223372036854775807, 18446744073709551615]"),
        )
        for not_int8 in ["z", "12 z", "128", "-129", "0.5", "0x", "0xfff", "-0xf0"]:
            yield fails(J(string_type, [not_int8]), pa.int8())
        for not_uint8 in ["256", "-1", "0.5", "0x", "0x3wa", "0x123"]:
            yield fails(J(string_type, [not_uint8]), pa.uint8())


@cpp_test("StringToFloating")
def _():
    # The C++ test repeats the cast under a French locale; Rust parsing has
    # no locale.
    for string_type in [pa.utf8(), pa.large_utf8()]:
        for float_type in [pa.float32(), pa.float64()]:
            yield check(J(string_type, '["0.1", null, "127.3", "1e3", "200.4", "0.5"]'), J(float_type, "[0.1, null, 127.3, 1000, 200.4, 0.5]"))
            yield fails(J(string_type, '["z"]'), pa.float32())


@cpp_test("StringToDecimal")
def _():
    for string_type in [pa.utf8(), pa.large_utf8()]:
        for decimal_type in DECIMALS_5_2:
            yield check(J(string_type, '["0.01", null, "127.32", "200.43", "0.54"]'), J(decimal_type, '["0.01", null, "127.32", "200.43", "0.54"]'))
            yield fails(J(string_type, '["z"]'), pa.decimal128(5, 2))


@cpp_test("StringToTimestamp")
def _():
    for string_type in [pa.utf8(), pa.large_utf8()]:
        strings = J(string_type, '["1970-01-01", null, "2000-02-29"]')
        yield check(strings, J(pa.timestamp("s"), "[0, null, 951782400]"))
        yield check(strings, J(pa.timestamp("us"), "[0, null, 951782400000000]"))
        for unit in UNITS:
            for not_ts in ["", "xxx"]:
                yield fails(J(string_type, [not_ts]), pa.timestamp(unit))
        zoned = J(string_type, '["2020-02-29T00:00:00Z", "2020-03-02T10:11:12+0102"]')
        mixed = J(string_type, '["2020-03-02T10:11:12+0102", "2020-02-29T00:00:00"]')
        yield fails(zoned, pa.timestamp("s"))
        yield fails(mixed, pa.timestamp("s"))
        yield fails(mixed, pa.timestamp("s", "UTC"))
        yield fails(strings, pa.timestamp("s", "UTC"))
        yield check(zoned, J(pa.timestamp("s", "UTC"), "[1582934400, 1583140152]"))
        yield check(zoned, J(pa.timestamp("s", "America/Phoenix"), "[1582934400, 1583140152]"))


@cpp_test("StringToDate")
def _():
    for string_type in [pa.utf8(), pa.large_utf8()]:
        strings = J(string_type, '["1970-01-01", null, "2000-02-29"]')
        yield check(strings, J(pa.date32(), "[0, null, 11016]"))
        yield check(strings, J(pa.date64(), "[0, null, 951782400000]"))
        for date_type in [pa.date32(), pa.date64()]:
            for not_date in ["", "2012-01-xx", "2012-01-01 09:00:00"]:
                yield fails(J(string_type, [not_date]), date_type)


BINARY_TYPES = [pa.binary(), pa.binary_view(), pa.large_binary()]
STRING_OF_BINARY = {pa.binary(): pa.utf8(), pa.binary_view(): pa.string_view(), pa.large_binary(): pa.large_utf8()}
BINARY_OF_STRING = {string: binary for binary, string in STRING_OF_BINARY.items()}
INVALID_UTF8 = [b"Hi", "olá mundo".encode(), "你好世界".encode(), b"", b"\xa0\xa1"]
FIXED_SIZE_INVALID_UTF8 = [b"Hi!", "lá".encode(), "你".encode(), b"   ", b"\xa0\xa1\xa2"]


def invalid_utf8(type, values=INVALID_UTF8):
    """C++ `InvalidUtf8` and `FixedSizeInvalidUtf8`: the bytes of `values`
    as `type`, string types included, whose last value is not UTF-8."""
    if type in BINARY_OF_STRING:
        return pa.array(values, BINARY_OF_STRING[type]).view(type)
    return pa.array(values, type)


@cpp_test("BinaryToString")
def _():
    for bin_type in BINARY_TYPES:
        for string_type in STRING_TYPES:
            yield check(pa.array([], bin_type), pa.array([], string_type))
            yield check(mask_nulls_at(invalid_utf8(bin_type), [4]), mask_nulls_at(invalid_utf8(string_type), [4]))
            yield fails(invalid_utf8(bin_type), string_type)
            yield excepted(invalid_utf8(bin_type), string_type, "allow invalid utf8", INVALID_UTF8_STRINGS, allow_invalid_utf8=True)
            yield excepted(invalid_utf8(bin_type), string_type, "unsafe", INVALID_UTF8_STRINGS, **UNSAFE)
    from_type = pa.binary(3)
    for string_type in STRING_TYPES:
        yield check(pa.array([], from_type), pa.array([], string_type))
        yield check(
            mask_nulls_at(invalid_utf8(from_type, FIXED_SIZE_INVALID_UTF8), [4]),
            mask_nulls_at(invalid_utf8(string_type, FIXED_SIZE_INVALID_UTF8), [4]),
        )
        yield fails(invalid_utf8(from_type, FIXED_SIZE_INVALID_UTF8), string_type)
        yield excepted(invalid_utf8(from_type, FIXED_SIZE_INVALID_UTF8), string_type, "allow invalid utf8", INVALID_UTF8_STRINGS, allow_invalid_utf8=True)
        yield excepted(invalid_utf8(from_type, FIXED_SIZE_INVALID_UTF8), string_type, "unsafe", INVALID_UTF8_STRINGS, **UNSAFE)


@cpp_test("BinaryOrStringToBinary")
def _():
    # The zero copy assertions are `cast::tests::conversion_casts_reuse_buffers`.
    for from_type in BASE_BINARY_AND_VIEW:
        for to_type in BINARY_TYPES:
            yield check(pa.array([], from_type), pa.array([], to_type))
            yield check(invalid_utf8(from_type), invalid_utf8(to_type))
            yield check(mask_nulls_at(invalid_utf8(from_type), [4]), mask_nulls_at(invalid_utf8(to_type), [4]))
    from_type = pa.binary(3)
    fixed = invalid_utf8(from_type, FIXED_SIZE_INVALID_UTF8)
    yield check(fixed, fixed)
    yield fails(fixed, pa.binary(5))
    for to_type in BINARY_TYPES:
        yield check(pa.array([], from_type), pa.array([], to_type))
        yield check(fixed, invalid_utf8(to_type, FIXED_SIZE_INVALID_UTF8))
        yield check(mask_nulls_at(fixed, [4]), mask_nulls_at(invalid_utf8(to_type, FIXED_SIZE_INVALID_UTF8), [4]))


@cpp_test("StringToString")
def _():
    short_input = '["foo", null, "bar", "baz", "quu"]'
    long_input = '["foofoofoofoofoo", null, "barbarbarbarbarbarbar", "bazbazbazbazbazbazbaz", "quuquuquuquuquuquuquuquuquu"]'
    combine_input = '["foo", null, "barbarbarbarbarbarbar", "baz", "quu"]'
    for from_type in STRING_TYPES:
        for to_type in STRING_TYPES:
            yield check(pa.array([], from_type), pa.array([], to_type))
            yield check(mask_nulls_at(invalid_utf8(from_type), [4]), mask_nulls_at(invalid_utf8(to_type), [4]))
            yield check(invalid_utf8(from_type), invalid_utf8(to_type), allow_invalid_utf8=True)
            for text in [short_input, long_input, combine_input]:
                yield check(J(from_type, text), J(to_type, text))


@cpp_test("StringToStringWithOffset")
def _():
    with_nulls = '["foo", null, "bar", null, "quu", "foo", "baz", "bar", null, "bar", "baz", null]'
    no_nulls = '["foo", "aa", "bar", "bb", "quu", "foo", "baz", "bar", "cc", "bar", "baz", "foo"]'
    for from_type in [pa.utf8(), pa.large_utf8()]:
        for to_type in [pa.utf8(), pa.large_utf8()]:
            for offset, length in [(3, 5), (8, 2), (10, 1), (12, 0)]:
                for text in [with_nulls, no_nulls]:
                    source, target = J(from_type, text), J(to_type, text)
                    yield check(source.slice(offset), target.slice(offset))
                    yield check(source.slice(offset, length), target.slice(offset, length))


@cpp_test("BinaryOrStringToFixedSizeBinary")
def _():
    for in_type in [pa.utf8(), pa.large_utf8(), pa.string_view(), pa.binary(), pa.binary_view(), pa.large_binary()]:
        valid = J(in_type, '["foo", null, "bar", "baz", "quu"]')
        yield check(valid, J(pa.binary(3), '["foo", null, "bar", "baz", "quu"]'))
        yield fails(J(in_type, '["foo", null, "bar", "baz", "quux"]'), pa.binary(3))
        yield fails(valid, pa.binary(5))
        yield check(pa.array([], in_type), pa.array([], pa.binary(3)))
        yield check(pa.array([], in_type), pa.array([], pa.binary(5)))


BINARY_OR_STRING_TYPES = [pa.utf8(), pa.string_view(), pa.large_utf8(), pa.binary(), pa.binary_view(), pa.large_binary()]


@cpp_test("FixedSizeBinaryToBinaryOrString")
def _():
    for out_type in BINARY_OR_STRING_TYPES:
        yield check(J(pa.binary(3), '["foo", null, "bar", "baz", "quu"]'), J(out_type, '["foo", null, "bar", "baz", "quu"]'))
        yield check(pa.array([], pa.binary(3)), pa.array([], out_type))


@cpp_test("FixedSizeBinaryToBinaryOrStringWithSlice")
def _():
    for out_type in BINARY_OR_STRING_TYPES:
        yield check(J(pa.binary(3), '["foo", null, "bar", "baz", "quu"]').slice(1, 3), J(out_type, '[null, "bar", "baz"]'))
        yield check(J(pa.binary(3), '["foo", "bar", "baz", "quu"]').slice(1, 3), J(out_type, '["bar", "baz", "quu"]'))


@cpp_test("IntToString")
def _():
    for string_type in STRING_TYPES:
        for type, values, strings in [
            (pa.int8(), "[0, 1, 127, -128, null]", '["0", "1", "127", "-128", null]'),
            (pa.uint8(), "[0, 1, 255, null]", '["0", "1", "255", null]'),
            (pa.int16(), "[0, 1, 32767, -32768, null]", '["0", "1", "32767", "-32768", null]'),
            (pa.uint16(), "[0, 1, 65535, null]", '["0", "1", "65535", null]'),
            (pa.int32(), "[0, 1, 2147483647, -2147483648, null]", '["0", "1", "2147483647", "-2147483648", null]'),
            (pa.uint32(), "[0, 1, 4294967295, null]", '["0", "1", "4294967295", null]'),
            (pa.int64(), "[0, 1, 9223372036854775807, -9223372036854775808, null]", '["0", "1", "9223372036854775807", "-9223372036854775808", null]'),
            (pa.uint64(), "[0, 1, 18446744073709551615, null]", '["0", "1", "18446744073709551615", null]'),
        ]:
            yield check(J(type, values), J(string_type, strings))


@cpp_test("FloatingToString")
def _():
    for float_type in [pa.float16(), pa.float32(), pa.float64()]:
        values = [0.0, -0.0, 1.5, float("-inf"), float("inf"), float("nan"), None]
        if float_type == pa.float16():
            values = [None if v is None else np.float16(v) for v in values]
        for string_type in STRING_TYPES:
            yield check(pa.array(values, float_type), J(string_type, '["0", "-0", "1.5", "-inf", "inf", "nan", null]'))


@cpp_test("BooleanToString")
def _():
    for string_type in STRING_TYPES:
        yield check(J(pa.bool_(), "[true, true, false, null]"), J(string_type, '["true", "true", "false", null]'))


@cpp_test("ListToPrimitive")
def _():
    yield unsupported(J(pa.list_(pa.int8()), "[[1, 2], [3, 4]]"), pa.uint8())
    yield unsupported(J(pa.list_(pa.binary()), '[["1", "2"], ["3", "4"]]'), pa.utf8())


LIST_FACTORIES = [pa.list_, pa.large_list]


def _list_to_list(value_types, text):
    for make_src in LIST_FACTORIES:
        for make_dest in LIST_FACTORIES:
            for src in value_types:
                for dest in value_types:
                    yield check(J(make_src(src), text), J(make_dest(dest), text))


@cpp_test("ListToList")
def _():
    yield from _list_to_list([pa.int32(), pa.float32(), pa.int64()], "[[0], [1], null, [2, 3, 4], [5, 6], null, [], [7], [8, 9]]")


@cpp_test("ListToListNoNulls")
def _():
    yield from _list_to_list([pa.int32(), pa.float32(), pa.int64()], "[[0], [1], [2, 3, 4], [5, 6], [], [7], [8, 9]]")


@cpp_test("ListToListOptionsPassthru")
def _():
    for make_src in LIST_FACTORIES:
        for make_dest in LIST_FACTORIES:
            list_int32 = J(make_src(pa.int32()), "[[87654321]]")
            yield fails(list_int32, make_dest(pa.int16()))
            yield check(list_int32, J(make_dest(pa.int16()), "[[32689]]"), allow_int_overflow=True)


def _fsl_to_fsl(value_types, text, tweaked, children_nulls=True):
    for src_value in value_types:
        for dest_value in value_types:
            src_type, dest_type = pa.list_(src_value, 2), pa.list_(dest_value, 2)
            src = J(src_type, text)
            yield check(src, J(dest_type, text))
            yield check(mask_nulls_at(src, [1]), J(dest_type, tweaked))
            child = "[1, 2, null, 4, 5, null]" if children_nulls else "[1, 2, 3, 4, 5, 6]"
            fsl = pa.FixedSizeListArray.from_arrays(J(src_value, child).slice(2), 2)
            expected = "[null, 4, 5, null]" if children_nulls else "[3, 4, 5, 6]"
            yield check(fsl, pa.FixedSizeListArray.from_arrays(J(dest_value, expected), 2))
            yield check(
                mask_nulls_at(fsl, [1]),
                J(dest_type, "[[null, 4], null]" if children_nulls else "[[3, 4], null]"),
            )
            yield type_error(src, pa.list_(dest_value, 3))


@cpp_test("FSLToFSL")
def _():
    yield from _fsl_to_fsl([pa.int32(), pa.float32(), pa.int64()], "[[0, 1], [2, 3], [null, 5], null]", "[[0, 1], null, [null, 5], null]")


@cpp_test("FSLToFSLNoNulls")
def _():
    yield from _fsl_to_fsl([pa.int32(), pa.float32(), pa.int64()], "[[0, 1], [2, 3], [4, 5]]", "[[0, 1], null, [4, 5]]", children_nulls=False)


def _options_pass_through(to):
    fsl_int32 = J(pa.list_(pa.int32(), 1), "[[87654321]]")
    yield fails(fsl_int32, to)
    # The C++ test checks the cast to `fixed_size_list<int16>[1]` whatever `to` is.
    yield check(fsl_int32, J(pa.list_(pa.int16(), 1), "[[32689]]"), allow_int_overflow=True)


@cpp_test("FSLToFSLOptionsPassThru")
def _():
    yield from _options_pass_through(pa.list_(pa.int16(), 1))


@cpp_test("FSLToList")
def _():
    nulls = "[[0, 1], [2, 3], [null, 5], null]"
    yield check(J(pa.list_(pa.int16(), 2), nulls), J(pa.list_(pa.int16()), nulls))
    yield check(J(pa.list_(pa.int16(), 2), nulls), J(pa.large_list(pa.int16()), nulls))
    yield check(J(pa.list_(pa.int16(), 2), nulls), J(pa.list_(pa.int32()), nulls))
    yield check(J(pa.list_(pa.int16(), 2), "[[0, 1], [2, 3], [4, 5]]"), J(pa.list_(pa.int32()), "[[0, 1], [2, 3], [4, 5]]"))
    nested = "[[[0, 1], [2, 3]], [[4, 5], null]]"
    yield check(J(pa.list_(pa.list_(pa.int16()), 2), nested), J(pa.list_(pa.list_(pa.int32())), nested))
    sliced = pa.FixedSizeListArray.from_arrays(J(pa.int32(), "[1, 2, null, 4, 5, null]").slice(2), 2)
    yield check(sliced, J(pa.list_(pa.int32()), "[[null, 4], [5, null]]"))
    yield from _options_pass_through(pa.list_(pa.int16()))


@cpp_test("ListToFSL")
def _():
    nulls = "[[0, 1], [2, 3], null, [null, 5], null]"
    yield check(J(pa.list_(pa.int16()), nulls), J(pa.list_(pa.int16(), 2), nulls))
    yield check(J(pa.large_list(pa.int16()), nulls), J(pa.list_(pa.int16(), 2), nulls))
    yield check(J(pa.list_(pa.int32()), nulls), J(pa.list_(pa.int16(), 2), nulls))
    yield check(J(pa.list_(pa.int32()), "[[0, 1], [2, 3], [4, 5]]"), J(pa.list_(pa.int16(), 2), "[[0, 1], [2, 3], [4, 5]]"))
    nested = "[[[0, 1], [2, 3]], [[4, 5], null]]"
    yield check(J(pa.list_(pa.list_(pa.int32())), nested), J(pa.list_(pa.list_(pa.int16()), 2), nested))
    sliced = pa.ListArray.from_arrays(J(pa.int32(), "[0, 2, 4]"), J(pa.int32(), "[1, 2, null, 4, 5, null]").slice(2))
    yield check(sliced, J(pa.list_(pa.int32(), 2), "[[null, 4], [5, null]]"))
    masked = mask_nulls_at(J(pa.list_(pa.int32()), "[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]"), [1, 2, 4])
    yield check(masked, J(pa.list_(pa.int32(), 2), "[[0, 1], null, null, [6, 7], null]"))
    yield from _options_pass_through(pa.list_(pa.int16()))
    yield fails(J(pa.list_(pa.int32()), "[[0, 1, 2], null, [3, 4]]"), pa.list_(pa.int32(), 3))


@cpp_test("CastMap")
def _():
    map_json = '[[["x", 1], ["y", 8], ["z", 9]], [["x", 6]], [["y", 36]]]'
    map_json_nullable = '[[["x", 1], ["y", null], ["z", 9]], null, [["y", 36]]]'
    src_type = pa.map_(pa.field("x", pa.utf8(), False), pa.field("y", pa.int64()))
    struct = pa.struct([pa.field("a", pa.utf8()), pa.field("b", pa.int64())])
    for dst_type in [
        pa.map_(pa.field("a", pa.utf8(), False), pa.field("b", pa.int64())),
        pa.map_(pa.large_utf8(), pa.field("y", pa.int32())),
        pa.list_(struct),
        pa.large_list(struct),
    ]:
        for text in [map_json, map_json_nullable]:
            yield check(J(src_type, text), J(dst_type, text))
    src_type = pa.map_(pa.utf8(), pa.field("x", pa.list_(pa.field("a", pa.int64()))))
    dst_type = pa.map_(pa.utf8(), pa.field("y", pa.list_(pa.field("b", pa.int64()))))
    nested = '[[["1", [1,2,3]]], [["2", [4,5,6]]]]'
    yield check(J(src_type, nested), J(dst_type, nested))
    three = pa.list_(pa.struct([pa.field("key", pa.int32()), pa.field("value", pa.int64()), pa.field("extra", pa.int64())]))
    yield type_error(J(src_type, nested), three)


def S(arrays, names, valid=None):
    """C++ `StructArray::Make(arrays, names, null_bitmap)`: the children are
    unchanged and `valid` is the struct's own validity."""
    mask = None if valid is None else pa.array([not v for v in valid])
    if names and isinstance(names[0], pa.Field):
        return pa.StructArray.from_arrays(arrays, fields=names, mask=mask)
    return pa.StructArray.from_arrays(arrays, names=names, mask=mask)


@cpp_test("StructToSameSizedAndNamedStruct")
def _():
    for src in NUMERIC:
        for dest in NUMERIC:
            a1, b1 = J(src, "[1, 2, 3, 4, null]"), J(src, "[null, 7, 8, 9, 0]")
            a2, b2 = J(dest, "[1, 2, 3, 4, null]"), J(dest, "[null, 7, 8, 9, 0]")
            yield check(S([a1, b1], ["a", "b"]), S([a2, b2], ["a", "b"]))
            valid = [False, True, False, True, False]
            yield check(S([a1, b1], ["a", "b"], valid), S([a2, b2], ["a", "b"], valid))


def _struct_subset(b, d, e, valid=None):
    """C++ `CheckStructToStructSubset` and, with `valid`,
    `CheckStructToStructSubsetWithNulls`, whose third destination is
    `{a, d, e}` rather than `{c, d, e}`."""
    for src in NUMERIC:
        for dest in NUMERIC:
            values = ["[1, 2, 5]", b, "[9, 11, 44]", d, e]
            a1, b1, c1, d1, e1 = (J(src, v) for v in values)
            a2, b2, c2, d2, e2 = (J(dest, v) for v in values)
            nulls = J(dest, "[null, null, null]")
            source = S([a1, b1, c1, d1, e1], ["a", "b", "c", "d", "e"], valid)
            third = (a2, "a") if valid else (c2, "c")
            for arrays, names in [
                ([a2], ["a"]),
                ([b2, c2], ["b", "c"]),
                ([third[0], d2, e2], [third[1], "d", "e"]),
                ([a2, b2, c2, e2], ["a", "b", "c", "e"]),
                ([a2, b2, c2, d2, e2], ["a", "b", "c", "d", "e"]),
                ([a2, d2, nulls], ["a", "d", "f"]),
                ([a2, c2, b2], ["a", "c", "b"]),
                ([a2, c2, d2, nulls], ["a", "c", "d", "a"]),
            ]:
                yield check(source, S(arrays, names, valid))
            yield type_error(source, pa.struct([pa.field("a", pa.int8()), pa.field("d", pa.int16()), pa.field("f", pa.int64(), False)]))
            last = pa.int8() if valid else pa.int64()
            fourth = pa.int64() if valid else pa.int32()
            yield type_error(source, pa.struct([pa.field("a", pa.int8()), pa.field("c", pa.int16()), pa.field("d", fourth), pa.field("a", last, False)]))
            duplicate = S([a1, b1, c1], ["a", "a", "a"], valid)
            for arrays in [[a2], [a2, b2], [a2, b2, c2], [a2, b2, c2, nulls]]:
                yield check(duplicate, S(arrays, ["a"] * len(arrays), valid))
            yield type_error(duplicate, pa.struct([pa.field("a", pa.int8())] * 3 + [pa.field("a", pa.int8(), False)]))


@cpp_test("StructToStructSubset")
def _():
    yield from _struct_subset("[3, 4, 7]", "[6, 51, 49]", "[19, 17, 74]")


@cpp_test("StructToStructSubsetWithNulls")
def _():
    yield from _struct_subset("[3, null, 7]", "[6, 51, null]", "[null, 17, 74]", [False, True, False])


@cpp_test("StructToSameSizedButDifferentNamedStruct")
def _():
    src = S([J(pa.int8(), "[1, 2]"), J(pa.int8(), "[3, 4]")], ["a", "b"])
    nulls = J(pa.int8(), "[null, null]")
    yield check(src, S([nulls, nulls], ["c", "d"]))
    yield type_error(src, pa.struct([pa.field("c", pa.int8(), False), pa.field("d", pa.int8())]))


@cpp_test("StructToBiggerStruct")
def _():
    src = S([J(pa.int8(), "[1, 2]"), J(pa.int8(), "[3, 4]")], ["a", "b"])
    yield type_error(src, pa.struct([pa.field("a", pa.int8()), pa.field("b", pa.int8()), pa.field("c", pa.int8(), False)]))
    yield type_error(src, pa.struct([pa.field("a", pa.int8()), pa.field("c", pa.int8(), False), pa.field("b", pa.int8())]))


@cpp_test("StructToBiggerNullableStruct")
def _():
    a, b, c = J(pa.int8(), "[1, 2]"), J(pa.int8(), "[3, 4]"), J(pa.int8(), "[null, null]")
    src = S([a, b], ["a", "b"])
    yield check(src, S([a, b, c], ["a", "b", "c"]))
    yield check(src, S([a, c, b], ["a", "c", "b"]))


@cpp_test("StructToDifferentNullabilityStruct")
def _():
    values = ["[11, 23, 56]", "[32, 46, 37]", "[95, 11, 44]"]
    src = S([J(pa.int8(), v) for v in values], [pa.field(n, pa.int8(), False) for n in "abc"])
    fields = [pa.field(n, pa.int64(), True) for n in "abc"]
    arrays = [J(pa.int64(), v) for v in values]
    yield check(src, S(arrays, fields))
    yield check(src, S([arrays[0], arrays[2]], [fields[0], fields[2]]))
    yield check(src, S([arrays[1]], [fields[1]]))

    src = S([J(pa.int8(), v) for v in ["[1, null, 5]", "[3, 4, null]", "[9, 11, 44]"]], [pa.field(n, pa.int8(), True) for n in "abc"])
    fields = [pa.field(n, pa.int64(), False) for n in "abc"]
    yield fails(src, pa.struct(fields))
    yield fails(src, pa.struct([fields[0], fields[2]]))
    yield check(src, S([J(pa.int64(), "[9, 11, 44]")], [fields[2]]))
    yield check(src.slice(0, 1), S([J(pa.int64(), v) for v in ["[1]", "[3]", "[9]"]], fields))
    yield fails(src.slice(1, 3), pa.struct(fields))


C_NUMERIC = [pa.uint8(), pa.int8(), pa.uint16(), pa.int16(), pa.uint32(), pa.int32(), pa.uint64(), pa.int64(), pa.float32(), pa.float64()]


@cpp_test("IdentityCasts")
def _():
    # The zero copy assertions are `cast::tests::conversion_casts_reuse_buffers`.
    identities = [(pa.null(), "[null, null, null]"), (pa.bool_(), "[false, true, null, false]")]
    identities += [(type, "[1, 2, null, 4]") for type in C_NUMERIC]
    identities += [(type, '["foo", "bar"]') for type in [pa.utf8(), pa.binary(), pa.string_view(), pa.binary_view(), pa.binary(3)]]
    identities += [
        (pa.list_(pa.int8()), "[[1, 2], [null], [], [3]]"),
        (pa.time32("ms"), "[1, 2, 3, 4]"),
        (pa.time64("us"), "[1, 2, 3, 4]"),
        (pa.date32(), "[1, 2, 3, 4]"),
        (pa.date64(), "[86400000, 0]"),
        (pa.timestamp("s"), "[1, 2, 3, 4]"),
        (pa.dictionary(pa.int8(), pa.int8()), "[1, 2, 3, 1, null, 3]"),
    ]
    for type, text in identities:
        array = J(type, text)
        yield check(array, array)


@cpp_test("EmptyCasts")
def _():
    def empty(type):
        # Python creates array with nullptr instead of 0-length (valid) buffers.
        return pa.Array.from_buffers(type, 0, [None, None])

    for numeric in C_NUMERIC:
        yield check(empty(pa.bool_()), pa.array([], numeric))
        yield check(empty(numeric), pa.array([], pa.bool_()))


@cpp_test("CastWithNoValidityBitmapButUnknownNullCount")
def _():
    values = J(pa.bool_(), "[true, true, false]")
    unknown = pa.Array.from_buffers(pa.bool_(), 3, values.buffers(), null_count=-1)
    yield check(unknown, J(pa.int8(), "[1, 1, 0]"))


@cpp_test("FromNull")
def _():
    for to_type in [pa.null()] + C_NUMERIC + [pa.date32(), pa.date64(), pa.binary(10), pa.binary(), pa.utf8()]:
        yield check(pa.nulls(10), pa.nulls(10, to_type))


@cpp_test("FromNullToDictionary")
def _():
    yield check(pa.nulls(10), pa.nulls(10, pa.dictionary(pa.int8(), pa.bool_())))


@cpp_test("FromDictionary")
def _():
    dictionaries = [pa.nulls(5)]
    dictionaries += [J(type, "[23, 12, 45, 12, null]") for type in C_NUMERIC]
    dictionaries += [J(type, '["foo", "bar", "baz", "foo", null]') for type in BASE_BINARY]
    for dictionary in dictionaries:
        for index_type in INTEGER:
            indices = J(index_type, "[4, 0, 1, 2, 0, 4, null, 2]")
            yield check(pa.DictionaryArray.from_arrays(indices, dictionary), dictionary.take(indices))
    for dictionary in dictionaries[1:]:
        no_nulls = dictionary.take(J(pa.int8(), "[0, 0, 1, 2, 0, 3, 3, 2]"))
        encoded = no_nulls.dictionary_encode()
        assert encoded.buffers()[0] is None
        yield check(encoded, no_nulls)


class _Extension(pa.ExtensionType):
    """The C++ test extension types `smallint` and `tinyint` of
    `cpp/src/arrow/testing/extension_type.h`: a name over an integer storage."""

    def __init__(self, storage, name):
        super().__init__(storage, name)

    def __arrow_ext_serialize__(self):
        return self.extension_name.encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls(storage_type, serialized.decode())


SMALLINT = _Extension(pa.int16(), "smallint")
TINYINT = _Extension(pa.int8(), "tinyint")


def smallint(text):
    """C++ `SmallintArrayFromJSON`."""
    return pa.ExtensionArray.from_storage(SMALLINT, J(pa.int16(), text))


def tinyint(text):
    """C++ `TinyintArrayFromJSON`."""
    return pa.ExtensionArray.from_storage(TINYINT, J(pa.int8(), text))


@cpp_test("ExtensionTypeToIntDowncast")
def _():
    yield check(smallint("[0, 100, 200, 1, 2]"), J(pa.int16(), "[0, 100, 200, 1, 2]"), "zero copy")
    yield check(smallint("[0, 100, 200, 1, 2]"), J(pa.uint8(), "[0, 100, 200, 1, 2]"))
    yield fails(smallint("[0, null, 256, 1, 3]"), pa.uint8(), "overflow")
    yield check(smallint("[0, null, 256, 1, 3]"), J(pa.uint8(), "[0, null, 0, 1, 3]"), "overflow", allow_int_overflow=True)
    yield fails(smallint("[0, null, -1, 1, 3]"), pa.uint8(), "underflow")
    yield check(smallint("[0, null, -1, 1, 3]"), J(pa.uint8(), "[0, null, 255, 1, 3]"), "underflow", allow_int_overflow=True)


@cpp_test("PrimitiveToExtension")
def _():
    yield check(J(pa.uint8(), "[0, 1, 3]"), smallint("[0, 1, 3]"))
    yield fails(J(pa.utf8(), '["hello"]'), SMALLINT)


@cpp_test("ExtensionDictToExtension")
def _():
    extension = smallint("[1, 2, 1]")
    encoded = pa.DictionaryArray.from_arrays(J(pa.int32(), "[0, 1, 0]"), extension)
    yield check(encoded, extension)


@cpp_test("IntToExtensionTypeDowncast")
def _():
    yield check(J(pa.uint8(), "[0, 100, 200, 1, 2]"), smallint("[0, 100, 200, 1, 2]"))
    yield fails(J(pa.int32(), "[0, null, 32768, 1, 3]"), SMALLINT, "overflow")
    yield check(J(pa.int32(), "[0, null, 32768, 1, 3]"), smallint("[0, null, -32768, 1, 3]"), "overflow", allow_int_overflow=True)
    yield fails(J(pa.int32(), "[0, null, -32769, 1, 3]"), SMALLINT, "underflow")
    yield check(J(pa.int32(), "[0, null, -32769, 1, 3]"), smallint("[0, null, 32767, 1, 3]"), "underflow", allow_int_overflow=True)
    # ASSERT_NOT_OK: Arrow C++ refuses an extension of another storage type
    # as a TypeError.
    yield type_error(tinyint("[0, 1, 3]"), SMALLINT, "different storage")


@cpp_test("DictTypeToAnotherDict")
def _():
    for in_type, out_type, text in [
        (pa.dictionary(pa.int8(), pa.int16()), pa.dictionary(pa.int8(), pa.int16()), "[1, 2, 3, 1, null, 3]"),
        (pa.dictionary(pa.int8(), pa.int16()), pa.dictionary(pa.int32(), pa.int64()), "[1, 2, 3, 1, null, 3]"),
        (pa.dictionary(pa.int8(), pa.int16()), pa.dictionary(pa.int32(), pa.float64()), "[1, 2, 3, 1, null, 3]"),
        (pa.dictionary(pa.int32(), pa.utf8()), pa.dictionary(pa.int8(), pa.utf8()), '["a", "b", "a", null]'),
    ]:
        array = J(in_type, text)
        yield check(array, array if in_type == out_type else J(out_type, text))
    array = J(pa.dictionary(pa.int32(), pa.int32()), "[1, 1000]")
    # The C++ test checks only that the unsafe cast succeeds with a valid result.
    yield attempt(array, pa.dictionary(pa.int8(), pa.int8()), **UNSAFE)
    yield fails(array, pa.dictionary(pa.int8(), pa.int8()))


def scalar(input, expected=None, to=None, expect=None, **options):
    """A scalar cast of the one value of `input`, as the C++ `Scalar` tests
    call the cast function. `expected` is the cast result, or `expect` the
    refusal category when it is `None`."""
    to = expected.type if expected is not None else to
    return Case(input, to, options, expected if expected is not None else expect, "scalar", variants="scalars")


@cpp_test("TestNullScalar.Cast")
def _():
    # The C++ test casts with Scalar::CastTo, since the cast function refuses a
    # null scalar; the cast function's null array casts are compared here.
    for to in [pa.int8(), pa.float64(), pa.date32(), pa.time32("s"), pa.timestamp("s"), pa.duration("s"), pa.utf8(), pa.large_binary(), pa.list_(pa.int32()), pa.struct([pa.field("f", pa.int32())]), pa.map_(pa.utf8(), pa.int32()), pa.decimal128(12, 2), pa.list_view(pa.int32()), pa.large_list(pa.int32()), pa.dense_union([pa.field("string", pa.utf8()), pa.field("number", pa.uint64())]), pa.sparse_union([pa.field("string", pa.utf8()), pa.field("number", pa.uint64())])]:
        yield attempt(pa.nulls(1), to)


@cpp_test("TestBooleanScalar.Cast")
def _():
    for b in [True, False]:
        value = pa.array([b])
        yield scalar(value, value)
        for to in [pa.int8(), pa.uint16(), pa.int32(), pa.uint64(), pa.float32(), pa.float64()]:
            yield scalar(value, pa.array([int(b)], to))
        yield scalar(value, pa.array(["true" if b else "false"]))


@cpp_test("TestScalar.IdentityCast")
def _():
    # `IntervalTypes()` day_time and month intervals, which PyArrow cannot
    # construct, are `cast::tests::conversion_casts_reuse_buffers` identities.
    values = [(pa.null(), [None]), (pa.bool_(), [True]), (pa.date32(), [1]), (pa.date64(), [86400000])]
    values += [(type, [3]) for type in INTEGER + [pa.float32(), pa.float64()]]
    values += [(type, ["foo"]) for type in BASE_BINARY + [pa.string_view(), pa.binary_view()]]
    values += [(type, [1]) for type in [pa.time32("s"), pa.time32("ms"), pa.time64("us"), pa.time64("ns")]]
    values += [(pa.timestamp(unit), [1]) for unit in UNITS] + [(pa.duration(unit), [1]) for unit in UNITS]
    values += [(pa.month_day_nano_interval(), [pa.MonthDayNano([1, 2, 3])])]
    values += [(pa.list_(pa.int32(), 20), [list(range(20))]), (pa.list_(pa.int32()), [[1, 2]]), (pa.large_list(pa.int32()), [[1, 2]])]
    for type, value in values:
        array = pa.array(value, type)
        yield scalar(array, array)


@cpp_test("TestDecimalScalar.Cast")
def _():
    for type in [pa.decimal32(3, 2), pa.decimal64(3, 2), pa.decimal128(3, 2), pa.decimal256(3, 2)]:
        yield scalar(decimal_unscaled(type, [314]), pa.array(["3.14"]))


@cpp_test("TestBinaryScalar.Cast")
def _():
    yield scalar(pa.array([b"test data"]), pa.array(["test data"]))


@cpp_test("TestFixedSizeBinaryScalar.Cast")
def _():
    yield scalar(pa.array([b"test data"], pa.binary(9)), pa.array(["test data"]))


@cpp_test("TestDateScalars.MakeScalar")
def _():
    for type in [pa.date32(), pa.date64()]:
        yield scalar(pa.array([5], type), pa.array([5], type))


@cpp_test("TestTimeScalars.MakeScalar")
def _():
    for type in [pa.time32("s"), pa.time32("ms"), pa.time64("us"), pa.time64("ns")]:
        yield scalar(pa.array([5], type), pa.array([5], type))


@cpp_test("TestTimestampScalars.Cast")
def _():
    for unit_in, unit_out, value, result in [("s", "ms", 1, 1000), ("s", "ns", 1, 1000000000), ("ns", "us", 1234, 1), ("us", "ms", 4567, 4)]:
        yield scalar(pa.array([value], pa.timestamp(unit_in)), pa.array([result], pa.timestamp(unit_out)), **UNSAFE)
    millis = pa.array([1024], pa.timestamp("ms"))
    yield scalar(millis, pa.array(["1970-01-01 00:00:01.024"]))
    yield scalar(millis, pa.array([1024], pa.int64()))
    day = 86400000
    yield scalar(pa.array([1024 * day + 3], pa.timestamp("ms")), pa.array([1024 * day], pa.date64()))


@cpp_test("TestNumericScalar.Cast")
def _():
    for type in [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(), pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.float32(), pa.float64(), pa.float16()]:
        for repr in ["0", "1", "3"]:
            value = pa.array([repr]).cast(type)
            for other_type in [pa.float32(), pa.int8(), pa.int64(), pa.uint32()]:
                other = pa.array([repr]).cast(other_type)
                yield scalar(value, other)
                yield scalar(other, value)
            yield scalar(pa.array([repr]), value)
            if pa.types.is_integer(type):
                yield scalar(value, pa.array([repr]))


def _list_scalar_casts(value, from_type, invalid):
    """C++ `CheckListCast` of the list scalar of `value` to a list, large list
    and fixed size list of its length, and `CheckListCastError` to `invalid`."""
    array = pa.array([value.to_pylist()], from_type)
    item = value.type
    for to in [pa.list_(item), pa.large_list(item), pa.list_(item, len(value))]:
        yield scalar(array, pa.array([value.to_pylist()], to))
    category = "TypeError" if pa.types.is_fixed_size_list(from_type) else "Invalid"
    yield scalar(array, to=invalid, expect=category)


@cpp_test("TestListLikeScalar.Cast")
def _():
    # These cases execute as scalars only, as the C++ test does: Arrow C++
    # 24.0.0 casts a list view array to a list with too short an offsets
    # buffer, and refuses its cast to a fixed size list of the list's size.
    value = J(pa.int16(), "[1, 2, null]")
    for from_type in [pa.list_(pa.int16()), pa.large_list(pa.int16()), pa.list_view(pa.int16()), pa.large_list_view(pa.int16()), pa.list_(pa.int16(), 3)]:
        yield from _list_scalar_casts(value, from_type, pa.list_(pa.int16(), 5))


@cpp_test("TestFixedSizeListScalar.Cast")
def _():
    yield from _list_scalar_casts(J(pa.int16(), "[1, 2, 5]"), pa.list_(pa.int16(), 3), pa.list_(pa.int16(), 4))


@cpp_test("TestMapScalar.Cast")
def _():
    key_value = pa.struct([pa.field("key", pa.utf8(), False), pa.field("value", pa.int8())])
    entries = [("a", 1), ("b", 2)]
    array = pa.array([entries], pa.map_(pa.utf8(), pa.int8()))
    value = [{"key": k, "value": v} for k, v in entries]
    for to in [pa.list_(key_value), pa.large_list(key_value), pa.list_(key_value, 2)]:
        yield scalar(array, pa.array([value], to))
    yield scalar(array, to=pa.list_(key_value, 5), expect="Invalid")


@cpp_test("TestDictionaryScalar.Cast")
def _():
    for index_type in INTEGER:
        for value_type in [pa.utf8(), pa.large_utf8(), pa.binary(), pa.large_binary()]:
            dictionary = J(value_type, '["alpha", null, "gamma"]')
            for i in range(len(dictionary)):
                # The C++ test asserts the decoded value; the dictionary
                # itself may differ.
                yield scalar(dictionary.slice(i, 1), to=pa.dictionary(index_type, value_type))


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


def safe_boundaries():
    """Beyond the C++ tests: safe casts at the edges the C++ tests do not
    reach, compared with the Arrow C++ outcome of the identical input."""
    two_63, two_64 = 9223372036854775808.0, 18446744073709551616.0
    for from_type in [pa.float32(), pa.float64()]:
        yield attempt(J(from_type, [two_63]), pa.int64(), "2^63 to int64")
        yield attempt(J(from_type, [two_63]), pa.int64(), "2^63 to int64, int overflow allowed", allow_int_overflow=True)
        yield attempt(J(from_type, [None, two_63]), pa.int64(), "2^63 after a null to int64")
        yield attempt(mask_nulls_at(J(from_type, [two_63, 1.0]), [0]), pa.int64(), "2^63 behind a null to int64")
        yield attempt(J(from_type, [-two_63, 0.0, None]), pa.int64(), "-2^63 to int64")
        yield attempt(J(from_type, [two_64]), pa.uint64(), "2^64 to uint64")
        yield attempt(J(from_type, [-1.0]), pa.uint64(), "-1 to uint64")
        yield attempt(J(from_type, [2147483648.0]), pa.int32(), "2^31 to int32")
    yield attempt(J(pa.float64(), [9223372036854774784.0, None]), pa.int64(), "largest float64 below 2^63 to int64")
    yield attempt(J(pa.float64(), [18446744073709549568.0, None]), pa.uint64(), "largest float64 below 2^64 to uint64")
    unknown_zone = pa.array([0, None, 1], pa.int64()).view(pa.timestamp("s", "not-a-timezone"))
    yield attempt(unknown_zone, pa.int64(), "timestamp with an unknown zone to int64")
    yield attempt(unknown_zone, pa.timestamp("ms", "not-a-timezone"), "timestamp with an unknown zone to timestamp")
    yield attempt(unknown_zone, pa.utf8(), "timestamp with an unknown zone to utf8")
    yield attempt(unknown_zone, pa.date32(), "timestamp with an unknown zone to date32")


_REGISTRY.append(("SafeBoundaries", safe_boundaries))


def extension_context():
    """Beyond the C++ tests: extension types nested in another type, and
    empty casts to and from an extension type, compared with the Arrow C++
    outcome of the identical input."""
    offsets = pa.array([0, 3], pa.int32())
    tiny = pa.ListArray.from_arrays(offsets, tinyint("[0, 1, 3]"))
    small = pa.ListArray.from_arrays(offsets, smallint("[0, 1, 3]"))
    for to in [pa.list_(SMALLINT), pa.large_list(SMALLINT), pa.list_(SMALLINT, 3)]:
        yield attempt(tiny, to, "list of tinyint to another extension")
        yield Case(tiny.slice(0, 0), to, {}, None, "empty list of tinyint to another extension", variants=False)
    yield attempt(small, pa.list_(SMALLINT), "list of smallint to the same extension")
    yield attempt(small, pa.list_(pa.int16()), "list of smallint to its storage")
    yield attempt(small, pa.list_(pa.int32()), "list of smallint to a wider integer")
    yield attempt(J(pa.list_(pa.int8()), "[[0, 1, 3]]"), pa.list_(SMALLINT), "list of int8 to an extension")
    yield attempt(pa.FixedSizeListArray.from_arrays(tinyint("[0, 1, 3]"), 3), pa.list_(SMALLINT), "fixed size list of tinyint to another extension")
    yield attempt(pa.StructArray.from_arrays([tinyint("[0, 1]")], ["x"]), pa.struct([("x", SMALLINT)]), "struct of tinyint to another extension")
    yield attempt(pa.StructArray.from_arrays([smallint("[0, 1]")], ["x"]), pa.struct([("x", SMALLINT)]), "struct of smallint to the same extension")
    entries = pa.MapArray.from_arrays(pa.array([0, 1], pa.int32()), pa.array(["a"]), tinyint("[1]"))
    yield attempt(entries, pa.map_(pa.utf8(), SMALLINT), "map of tinyint to another extension")
    yield attempt(entries, pa.list_(pa.struct([("k", pa.utf8()), ("v", SMALLINT)])), "map of tinyint to a list of another extension")
    struct = pa.struct([("x", pa.int8())])
    yield attempt(J(struct, "[[1]]"), SMALLINT, "struct to an extension")
    yield attempt(J(struct, "[[1]]"), pa.int16(), "struct to the storage of an extension")
    for from_type in [*REGISTRATION_TYPES, TINYINT]:
        yield Case(pa.nulls(0, from_type), SMALLINT, {}, None, f"empty {from_type} to an extension", variants=False)
        yield Case(pa.nulls(0, from_type), pa.int16(), {}, None, f"empty {from_type} to the storage of an extension", variants=False)
    for to in REGISTRATION_TYPES:
        yield Case(smallint("[]"), to, {}, None, f"empty extension to {to}", variants=False)
        yield attempt(smallint("[1]"), to, f"extension to {to}")


_REGISTRY.append(("ExtensionContext", extension_context))


def cases():
    """Every `(C++ test, case number, Case)` of the corpus."""
    for name, function in _REGISTRY:
        for number, case in enumerate(function()):
            yield name, number, case


def unmapped_tests():
    translated = {name for name, _ in _REGISTRY}
    return [
        name
        for name in CPP_TESTS + SCALAR_CPP_TESTS
        if name not in translated and name not in MAPPED_ELSEWHERE
    ]
