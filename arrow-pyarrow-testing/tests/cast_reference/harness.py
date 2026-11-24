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

"""Runs the cast reference corpus through Arrow C++ (PyArrow) for
`tests/cast_reference.rs`, which runs the identical inputs through
`arrow_cast::cast_with_conversion_options` and compares the outcomes.

Each case is executed in the variants Arrow C++ `CheckScalar` uses in
`cpp/src/arrow/compute/kernels/test_util_internal.cc`: the whole array, every
element as a scalar, the three thirds of the array, an empty slice, and the
array split into chunks of a chunked array.
"""

import pyarrow as pa
import pyarrow.compute as pc

import corpus

FLAGS = (
    "allow_int_overflow",
    "allow_time_truncate",
    "allow_time_overflow",
    "allow_decimal_truncate",
    "allow_float_truncate",
    "allow_invalid_utf8",
)


def exportable(array):
    """`array` with an empty buffer in place of each null data buffer an Arrow
    C++ cast to a binary view type leaves for a chunk whose values are all
    inline, which the C Data Interface export dereferences. Buffer indices and
    values are unchanged."""
    if pa.types.is_string_view(array.type) or pa.types.is_binary_view(array.type):
        buffers = array.buffers()
        data = [pa.py_buffer(b"") if buffer is None else buffer for buffer in buffers[2:]]
        return pa.Array.from_buffers(
            array.type, len(array), buffers[:2] + data, array.null_count, array.offset
        )
    return array


def outcome(run):
    """`("ok", value)` or `(status category, message)` of one Arrow C++ call."""
    try:
        return ("ok", exportable(run()))
    except pa.ArrowNotImplementedError as error:
        return ("NotImplemented", str(error))
    except pa.ArrowInvalid as error:
        return ("Invalid", str(error))
    except pa.ArrowTypeError as error:
        return ("TypeError", str(error))
    except pa.ArrowException as error:
        return (type(error).__name__, str(error))


def cast_options(case):
    return pc.CastOptions(case.to, **case.options)


def as_scalar_array(scalar):
    """The length one array Arrow C++ `MakeArrayFromScalar` executes a scalar as."""
    return pa.repeat(scalar, 1)


def variants(case):
    """Yield `(variant, input array, reference outcome)` for every execution of `case`."""
    array = case.input
    options = cast_options(case)
    yield ("array", array, outcome(lambda: pc.cast(array, options=options)))
    if not case.variants:
        return
    for index in range(len(array)):
        scalar = array[index]
        yield (
            f"scalar[{index}]",
            as_scalar_array(scalar),
            outcome(lambda: as_scalar_array(scalar.cast(options=options))),
        )
    third = len(array) // 3
    slices = []
    if third > 0:
        slices = [(0, third), (third, third), (2 * third, len(array) - 2 * third)]
    for offset, length in slices + [(0, 0)]:
        sliced = array.slice(offset, length)
        yield (
            f"slice[{offset}:{offset + length}]",
            sliced,
            outcome(lambda: pc.cast(sliced, options=options)),
        )
    if third > 0:
        chunks = [array.slice(0, third), array.slice(third)]
        chunked = pa.chunked_array(chunks, type=array.type)
        result = outcome(lambda: pc.cast(chunked, options=options).combine_chunks())
        start = 0
        for index, chunk in enumerate(chunks):
            if result[0] == "ok":
                chunk_outcome = ("ok", result[1].slice(start, len(chunk)))
            else:
                # A refused chunked cast does not say which chunk refused.
                chunk_outcome = outcome(lambda: pc.cast(chunk, options=options))
            start += len(chunk)
            yield (f"chunk[{index}]", chunk, chunk_outcome)


def translation_error(case, reference):
    """Why the reference outcome disagrees with the C++ test's expectation, if it does."""
    kind, value = reference
    if case.expect is None:
        return None
    if case.expect == "supported":
        if kind == "NotImplemented":
            return f"C++ test expects a supported cast, Arrow C++ returned {kind}: {value}"
        return None
    if isinstance(case.expect, str):
        if kind != case.expect:
            return f"C++ test expects {case.expect}, Arrow C++ returned {kind}: {value}"
        return None
    if kind != "ok":
        return f"C++ test expects success, Arrow C++ returned {kind}: {value}"
    if not value.equals(case.expect) or value.type != case.expect.type:
        return f"C++ test expects {case.expect!r}, Arrow C++ returned {value!r}"
    return None


def executions():
    """Every execution of the corpus as a tuple of the C++ test, case identity,
    variant, input, target type, option flags, reference status, reference
    message or result, any disagreement with the C++ test's expectation, and
    the reason Rust is expected to refuse as `Invalid` where Arrow C++
    converts, if it is."""
    for test, number, case in corpus.cases():
        flags = {flag: bool(case.options.get(flag, False)) for flag in FLAGS}
        for variant, array, reference in variants(case):
            kind, value = reference
            translation = translation_error(case, reference) if variant == "array" else None
            yield (
                test,
                f"{test}#{number} {case.label}",
                variant,
                array,
                case.to,
                flags,
                kind,
                value if kind == "ok" else None,
                str(value) if kind != "ok" else "",
                translation,
                case.divergence,
            )


def unmapped_tests():
    """C++ cast tests the corpus neither translates nor records as mapped elsewhere."""
    return corpus.unmapped_tests()
