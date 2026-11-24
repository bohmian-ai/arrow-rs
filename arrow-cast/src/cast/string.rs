// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::cast::*;
use arrow_buffer::NullBuffer;

pub(crate) fn value_to_string<O: OffsetSizeTrait>(
    array: &dyn Array,
    options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let mut builder = GenericStringBuilder::<O>::new();
    let formatter = ArrayFormatter::try_new(array, &options.format_options)?;
    let nulls = array.nulls();
    for i in 0..array.len() {
        match nulls.is_some_and(|x| x.is_null(i)) {
            true => builder.append_null(),
            false => {
                formatter.value(i).write(&mut builder)?;
                // tell the builder the row is finished
                builder.append_value("");
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

pub(crate) fn value_to_string_view(
    array: &dyn Array,
    options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let mut builder = StringViewBuilder::with_capacity(array.len());
    let formatter = ArrayFormatter::try_new(array, &options.format_options)?;
    let nulls = array.nulls();
    // buffer to avoid reallocating on each value
    // TODO: replace with write to builder after https://github.com/apache/arrow-rs/issues/6373
    let mut buffer = String::new();
    for i in 0..array.len() {
        match nulls.is_some_and(|x| x.is_null(i)) {
            true => builder.append_null(),
            false => {
                // write to buffer first and then copy into target array
                buffer.clear();
                formatter.value(i).write(&mut buffer)?;
                builder.append_value(&buffer)
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Parse UTF-8
pub(crate) fn parse_string<P: Parser, O: OffsetSizeTrait>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let string_array = array.as_string::<O>();
    parse_string_iter::<P, _, _>(string_array.iter(), cast_options, || {
        string_array.nulls().cloned()
    })
}

/// Parse UTF-8 View
pub(crate) fn parse_string_view<P: Parser>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let string_view_array = array.as_string_view();
    parse_string_iter::<P, _, _>(string_view_array.iter(), cast_options, || {
        string_view_array.nulls().cloned()
    })
}

fn parse_string_iter<
    'a,
    P: Parser,
    I: Iterator<Item = Option<&'a str>>,
    F: FnOnce() -> Option<NullBuffer>,
>(
    iter: I,
    cast_options: &CastOptions,
    nulls: F,
) -> Result<ArrayRef, ArrowError> {
    let array = if cast_options.safe {
        let iter = iter.map(|x| x.and_then(P::parse));

        // Benefit:
        //     20% performance improvement
        // Soundness:
        //     The iterator is trustedLen because it comes from an `StringArray`.
        unsafe { PrimitiveArray::<P>::from_trusted_len_iter(iter) }
    } else {
        let v = iter
            .map(|x| match x {
                Some(v) => P::parse(v).ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "Cannot cast string '{v}' to value of {} type",
                        P::DATA_TYPE
                    ))
                }),
                None => Ok(P::Native::default()),
            })
            .collect::<Result<Vec<_>, ArrowError>>()?;
        PrimitiveArray::try_new(v.into(), nulls())?
    };

    Ok(Arc::new(array) as ArrayRef)
}

/// Casts generic string arrays to an ArrowTimestampType (TimeStampNanosecondArray, etc.)
pub(crate) fn cast_string_to_timestamp<O: OffsetSizeTrait, T: ArrowTimestampType>(
    array: &dyn Array,
    to_tz: Option<&Arc<str>>,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_string::<O>();
    let out: PrimitiveArray<T> = match to_tz {
        Some(tz) => {
            let tz: Tz = tz.as_ref().parse()?;
            cast_string_to_timestamp_impl(array.iter(), &tz, cast_options)?
        }
        None => cast_string_to_timestamp_impl(array.iter(), &Utc, cast_options)?,
    };
    Ok(Arc::new(out.with_timezone_opt(to_tz.cloned())))
}

/// Casts string view arrays to an ArrowTimestampType (TimeStampNanosecondArray, etc.)
pub(crate) fn cast_view_to_timestamp<T: ArrowTimestampType>(
    array: &dyn Array,
    to_tz: Option<&Arc<str>>,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_string_view();
    let out: PrimitiveArray<T> = match to_tz {
        Some(tz) => {
            let tz: Tz = tz.as_ref().parse()?;
            cast_string_to_timestamp_impl(array.iter(), &tz, cast_options)?
        }
        None => cast_string_to_timestamp_impl(array.iter(), &Utc, cast_options)?,
    };
    Ok(Arc::new(out.with_timezone_opt(to_tz.cloned())))
}

fn cast_string_to_timestamp_impl<
    'a,
    I: Iterator<Item = Option<&'a str>>,
    T: ArrowTimestampType,
    Tz: TimeZone,
>(
    iter: I,
    tz: &Tz,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError> {
    if cast_options.safe {
        let iter = iter.map(|v| {
            let naive = string_to_datetime(tz, v?).ok()?.naive_utc();
            T::from_naive_datetime(naive, None)
        });
        // Benefit:
        //     20% performance improvement
        // Soundness:
        //     The iterator is trustedLen because it comes from an `StringArray`.

        Ok(unsafe { PrimitiveArray::from_trusted_len_iter(iter) })
    } else {
        let vec = iter
            .map(|v| {
                v.map(|v| {
                    let naive = string_to_datetime(tz, v)?.naive_utc();
                    T::from_naive_datetime(naive, None).ok_or_else(|| match T::UNIT {
                        TimeUnit::Nanosecond => ArrowError::CastError(format!(
                            "Overflow converting {naive} to Nanosecond. The dates that can be represented as nanoseconds have to be between 1677-09-21T00:12:44.0 and 2262-04-11T23:47:16.854775804"
                        )),
                        _ => ArrowError::CastError(format!(
                            "Overflow converting {naive} to {:?}",
                            T::UNIT
                        ))
                    })
                })
                    .transpose()
            })
            .collect::<Result<Vec<Option<i64>>, _>>()?;

        // Benefit:
        //     20% performance improvement
        // Soundness:
        //     The iterator is trustedLen because it comes from an `StringArray`.
        Ok(unsafe { PrimitiveArray::from_trusted_len_iter(vec.iter()) })
    }
}

pub(crate) fn cast_string_to_interval<Offset, F, ArrowType>(
    array: &dyn Array,
    cast_options: &CastOptions,
    parse_function: F,
) -> Result<ArrayRef, ArrowError>
where
    Offset: OffsetSizeTrait,
    ArrowType: ArrowPrimitiveType,
    F: Fn(&str) -> Result<ArrowType::Native, ArrowError> + Copy,
{
    let string_array = array
        .as_any()
        .downcast_ref::<GenericStringArray<Offset>>()
        .unwrap();
    cast_string_to_interval_impl::<_, ArrowType, F>(
        string_array.iter(),
        cast_options,
        parse_function,
    )
}

pub(crate) fn cast_string_to_year_month_interval<Offset: OffsetSizeTrait>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_string_to_interval::<Offset, _, IntervalYearMonthType>(
        array,
        cast_options,
        parse_interval_year_month,
    )
}

pub(crate) fn cast_string_to_day_time_interval<Offset: OffsetSizeTrait>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_string_to_interval::<Offset, _, IntervalDayTimeType>(
        array,
        cast_options,
        parse_interval_day_time,
    )
}

pub(crate) fn cast_string_to_month_day_nano_interval<Offset: OffsetSizeTrait>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_string_to_interval::<Offset, _, IntervalMonthDayNanoType>(
        array,
        cast_options,
        parse_interval_month_day_nano,
    )
}

pub(crate) fn cast_view_to_interval<F, ArrowType>(
    array: &dyn Array,
    cast_options: &CastOptions,
    parse_function: F,
) -> Result<ArrayRef, ArrowError>
where
    ArrowType: ArrowPrimitiveType,
    F: Fn(&str) -> Result<ArrowType::Native, ArrowError> + Copy,
{
    let string_view_array = array.as_any().downcast_ref::<StringViewArray>().unwrap();
    cast_string_to_interval_impl::<_, ArrowType, F>(
        string_view_array.iter(),
        cast_options,
        parse_function,
    )
}

pub(crate) fn cast_view_to_year_month_interval(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_view_to_interval::<_, IntervalYearMonthType>(
        array,
        cast_options,
        parse_interval_year_month,
    )
}

pub(crate) fn cast_view_to_day_time_interval(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_view_to_interval::<_, IntervalDayTimeType>(array, cast_options, parse_interval_day_time)
}

pub(crate) fn cast_view_to_month_day_nano_interval(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    cast_view_to_interval::<_, IntervalMonthDayNanoType>(
        array,
        cast_options,
        parse_interval_month_day_nano,
    )
}

fn cast_string_to_interval_impl<'a, I, ArrowType, F>(
    iter: I,
    cast_options: &CastOptions,
    parse_function: F,
) -> Result<ArrayRef, ArrowError>
where
    I: Iterator<Item = Option<&'a str>>,
    ArrowType: ArrowPrimitiveType,
    F: Fn(&str) -> Result<ArrowType::Native, ArrowError> + Copy,
{
    let interval_array = if cast_options.safe {
        let iter = iter.map(|v| parse_function(v?).ok());

        // Benefit:
        //     20% performance improvement
        // Soundness:
        //     The iterator is trustedLen because it comes from an `StringArray`.
        unsafe { PrimitiveArray::<ArrowType>::from_trusted_len_iter(iter) }
    } else {
        let vec = iter
            .map(|v| v.map(parse_function).transpose())
            .collect::<Result<Vec<_>, ArrowError>>()?;

        // Benefit:
        //     20% performance improvement
        // Soundness:
        //     The iterator is trustedLen because it comes from an `StringArray`.
        unsafe { PrimitiveArray::<ArrowType>::from_trusted_len_iter(vec) }
    };
    Ok(Arc::new(interval_array) as ArrayRef)
}

/// A specified helper to cast from `GenericBinaryArray` to `GenericStringArray` when they have same
/// offset size so re-encoding offset is unnecessary.
fn extend_valid_utf8<'a, B, I>(builder: &mut B, iter: I)
where
    B: Extend<Option<&'a str>>,
    I: Iterator<Item = Option<&'a [u8]>>,
{
    builder.extend(iter.map(|value| std::str::from_utf8(value?).ok()));
}

pub(crate) fn cast_binary_to_string<O: OffsetSizeTrait>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let array = array
        .as_any()
        .downcast_ref::<GenericByteArray<GenericBinaryType<O>>>()
        .unwrap();

    match GenericStringArray::<O>::try_from_binary(array.clone()) {
        Ok(a) => Ok(Arc::new(a)),
        Err(e) => match cast_options.safe {
            true => {
                // Fallback to slow method to convert invalid sequences to nulls
                let mut builder =
                    GenericStringBuilder::<O>::with_capacity(array.len(), array.value_data().len());

                extend_valid_utf8(&mut builder, array.iter());
                Ok(Arc::new(builder.finish()))
            }
            false => Err(e),
        },
    }
}

pub(crate) fn cast_binary_view_to_string_view(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_binary_view();

    match array.clone().to_string_view() {
        Ok(result) => Ok(Arc::new(result)),
        Err(error) => match cast_options.safe {
            true => {
                let mut builder = StringViewBuilder::with_capacity(array.len());
                extend_valid_utf8(&mut builder, array.iter());
                Ok(Arc::new(builder.finish()))
            }
            false => Err(error),
        },
    }
}

/// Casts string to boolean
fn cast_string_to_boolean<'a, StrArray>(
    array: &StrArray,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    StrArray: StringArrayType<'a>,
{
    let output_array = array
        .iter()
        .map(|value| match value {
            Some(value) => cast_single_string_to_boolean(value, cast_options),
            None => Ok(None),
        })
        .collect::<Result<BooleanArray, _>>()?;

    Ok(Arc::new(output_array))
}

#[inline]
fn cast_single_string_to_boolean(
    value: &str,
    cast_options: &CastOptions,
) -> Result<Option<bool>, ArrowError> {
    match value.to_ascii_lowercase().trim() {
        "t" | "tr" | "tru" | "true" | "y" | "ye" | "yes" | "on" | "1" => Ok(Some(true)),
        "f" | "fa" | "fal" | "fals" | "false" | "n" | "no" | "of" | "off" | "0" => Ok(Some(false)),
        invalid_value => match cast_options.safe {
            true => Ok(None),
            false => Err(ArrowError::CastError(format!(
                "Cannot cast value '{invalid_value}' to value of Boolean type",
            ))),
        },
    }
}

/// Cast a single string to boolean with default cast option(safe=true).
pub fn cast_single_string_to_boolean_default(value: &str) -> Option<bool> {
    cast_single_string_to_boolean(value, &CastOptions::default())
        .ok()
        .flatten()
}

pub(crate) fn cast_utf8_to_boolean<OffsetSize>(
    from: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    OffsetSize: OffsetSizeTrait,
{
    let array = from
        .as_any()
        .downcast_ref::<GenericStringArray<OffsetSize>>()
        .unwrap();

    cast_string_to_boolean(&array, cast_options)
}

pub(crate) fn cast_utf8view_to_boolean(
    from: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    let array = from.as_any().downcast_ref::<StringViewArray>().unwrap();

    cast_string_to_boolean(&array, cast_options)
}

/// Parses every value of a string or binary array to `to_type` as Arrow C++
/// `ParseValue` does: integers are an optional `-` and decimal digits or a
/// `0x` hexadecimal literal of at most the type's width, floats follow
/// `fast_float` with an optional leading `+`, and booleans are `0`, `1`,
/// `true` or `false` in any case. Binary values are parsed as bytes.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first value that does not parse.
pub(crate) fn conversion_parse(
    array: &dyn Array,
    to_type: &DataType,
) -> Result<ArrayRef, ArrowError> {
    use DataType::*;
    match to_type {
        Boolean => Ok(Arc::new(
            parsed(array, to_type, parse_cpp_boolean).collect::<Result<BooleanArray, _>>()?,
        )),
        Int8 => parse_primitive::<Int8Type>(array, parse_cpp_integer),
        Int16 => parse_primitive::<Int16Type>(array, parse_cpp_integer),
        Int32 => parse_primitive::<Int32Type>(array, parse_cpp_integer),
        Int64 => parse_primitive::<Int64Type>(array, parse_cpp_integer),
        UInt8 => parse_primitive::<UInt8Type>(array, parse_cpp_integer),
        UInt16 => parse_primitive::<UInt16Type>(array, parse_cpp_integer),
        UInt32 => parse_primitive::<UInt32Type>(array, parse_cpp_integer),
        UInt64 => parse_primitive::<UInt64Type>(array, parse_cpp_integer),
        Float16 => parse_primitive::<Float16Type>(array, |s| {
            parse_cpp_float::<f32>(s).map(half::f16::from_f32)
        }),
        Float32 => parse_primitive::<Float32Type>(array, parse_cpp_float::<f32>),
        Float64 => parse_primitive::<Float64Type>(array, parse_cpp_float::<f64>),
        _ => unreachable!("only numeric and boolean targets are parsed"),
    }
}

/// Copies every value of a string or binary array into a fixed size binary
/// array of `width`, as Arrow C++ does.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first value whose length is not
/// `width`.
pub(crate) fn conversion_to_fixed_size_binary(
    array: &dyn Array,
    width: i32,
) -> Result<ArrayRef, ArrowError> {
    let mut builder = FixedSizeBinaryBuilder::with_capacity(array.len(), width);
    for value in byte_values(array) {
        match value {
            Some(bytes) => builder.append_value(bytes).map_err(|_| {
                ArrowError::CastError(format!(
                    "Failed casting from {} to FixedSizeBinary({width}): widths must match",
                    array.data_type()
                ))
            })?,
            None => builder.append_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// The values of a string or binary array as bytes.
fn byte_values(array: &dyn Array) -> Box<dyn Iterator<Item = Option<&[u8]>> + '_> {
    match array.data_type() {
        DataType::Utf8 => Box::new(
            array
                .as_string::<i32>()
                .iter()
                .map(|v| v.map(str::as_bytes)),
        ),
        DataType::LargeUtf8 => Box::new(
            array
                .as_string::<i64>()
                .iter()
                .map(|v| v.map(str::as_bytes)),
        ),
        DataType::Utf8View => Box::new(array.as_string_view().iter().map(|v| v.map(str::as_bytes))),
        DataType::Binary => Box::new(array.as_binary::<i32>().iter()),
        DataType::LargeBinary => Box::new(array.as_binary::<i64>().iter()),
        DataType::BinaryView => Box::new(array.as_binary_view().iter()),
        other => unreachable!("{other} is not a string or binary type"),
    }
}

/// Parses each non-null value of `array` with `parse`, refusing the first
/// value that does not parse.
fn parsed<'a, N: 'a>(
    array: &'a dyn Array,
    to_type: &'a DataType,
    parse: impl Fn(&[u8]) -> Option<N> + 'a,
) -> impl Iterator<Item = Result<Option<N>, ArrowError>> + 'a {
    byte_values(array).map(move |value| {
        value
            .map(|bytes| {
                parse(bytes).ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "Failed to parse string: '{}' as a scalar of type {to_type}",
                        String::from_utf8_lossy(bytes),
                    ))
                })
            })
            .transpose()
    })
}

fn parse_primitive<T: ArrowPrimitiveType>(
    array: &dyn Array,
    parse: impl Fn(&[u8]) -> Option<T::Native>,
) -> Result<ArrayRef, ArrowError> {
    let values = parsed(array, &T::DATA_TYPE, parse).collect::<Result<PrimitiveArray<T>, _>>()?;
    Ok(Arc::new(values))
}

fn parse_cpp_boolean(s: &[u8]) -> Option<bool> {
    match s {
        b"0" => Some(false),
        b"1" => Some(true),
        _ if s.eq_ignore_ascii_case(b"true") => Some(true),
        _ if s.eq_ignore_ascii_case(b"false") => Some(false),
        _ => None,
    }
}

/// Arrow C++ integer parsing: a `0x` literal of at most twice the type's
/// byte width in hexadecimal digits, whose bits are reinterpreted for a
/// signed type, or an optional `-` and decimal digits within range.
fn parse_cpp_integer<N: NumCast>(s: &[u8]) -> Option<N> {
    let bits = 8 * std::mem::size_of::<N>() as u32;
    let signed = <N as NumCast>::from(-1).is_some();
    if s.len() > 2 && s[0] == b'0' && (s[1] == b'x' || s[1] == b'X') {
        let digits = &s[2..];
        if digits.len() > (bits / 4) as usize {
            return None;
        }
        let mut value: u128 = 0;
        for digit in digits {
            value = (value << 4) | (*digit as char).to_digit(16)? as u128;
        }
        let value = if signed && value >> (bits - 1) == 1 {
            value as i128 - (1i128 << bits)
        } else {
            value as i128
        };
        return <N as NumCast>::from(value);
    }
    let (negative, digits) = match s {
        [] | [b'-'] => return None,
        [b'-', rest @ ..] if signed => (true, rest),
        _ => (false, s),
    };
    let mut value: i128 = 0;
    for digit in digits {
        if !digit.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((digit - b'0') as i128)?;
    }
    <N as NumCast>::from(if negative { -value } else { value })
}

/// Arrow C++ float parsing with `fast_float`: the general format with an
/// optional leading `+`, `inf`, `infinity` and `nan` in any case, an
/// out-of-range value saturating, and no surrounding characters.
fn parse_cpp_float<F: std::str::FromStr>(s: &[u8]) -> Option<F> {
    std::str::from_utf8(s).ok()?.parse().ok()
}
