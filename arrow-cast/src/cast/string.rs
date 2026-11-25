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
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let array = array
        .as_any()
        .downcast_ref::<GenericByteArray<GenericBinaryType<O>>>()
        .unwrap();

    match GenericStringArray::<O>::try_from_binary(array.clone()) {
        Ok(a) => Ok(Arc::new(a)),
        Err(e) => {
            if cast_options.conversion.is_some() {
                refuse_invalid_utf8(array.iter())?;
            } else if !cast_options.safe {
                return Err(e);
            }
            // Fallback to slow method to convert invalid sequences to nulls
            let mut builder =
                GenericStringBuilder::<O>::with_capacity(array.len(), array.value_data().len());

            extend_valid_utf8(&mut builder, array.iter());
            Ok(Arc::new(builder.finish()))
        }
    }
}

/// Refuses the first non-null value that is not UTF-8, as Arrow C++ validates
/// a binary to string cast. The bytes of null values are not checked: a
/// caller rebuilds the array without them, since a Rust string array must
/// hold UTF-8 in every value range.
///
/// # Errors
///
/// [`ArrowError::CastError`] naming the first invalid value.
fn refuse_invalid_utf8<'a>(
    values: impl Iterator<Item = Option<&'a [u8]>>,
) -> Result<(), ArrowError> {
    for (index, value) in values.enumerate() {
        if let Some(Err(error)) = value.map(std::str::from_utf8) {
            return Err(ArrowError::CastError(format!(
                "Invalid UTF8 payload at index {index}: {error}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn cast_binary_view_to_string_view(
    array: &dyn Array,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_binary_view();

    match array.clone().to_string_view() {
        Ok(result) => Ok(Arc::new(result)),
        Err(error) => {
            if cast_options.conversion.is_some() {
                refuse_invalid_utf8(array.iter())?;
            } else if !cast_options.safe {
                return Err(error);
            }
            let mut builder = StringViewBuilder::with_capacity(array.len());
            extend_valid_utf8(&mut builder, array.iter());
            Ok(Arc::new(builder.finish()))
        }
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
pub(crate) fn byte_values(array: &dyn Array) -> Box<dyn Iterator<Item = Option<&[u8]>> + '_> {
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
pub(crate) fn parse_cpp_integer<N: NumCast>(s: &[u8]) -> Option<N> {
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
/// optional leading `+`, `inf`, `infinity` and `nan` in any case, `nan`
/// followed by a parenthesized sequence of ASCII letters, digits and
/// underscores, which reads as `nan`, an out-of-range value saturating, and
/// no surrounding characters.
fn parse_cpp_float<F: std::str::FromStr>(s: &[u8]) -> Option<F> {
    let s = std::str::from_utf8(s).ok()?;
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    let payload = unsigned
        .get(..4)
        .filter(|nan| nan.eq_ignore_ascii_case("nan("))
        .and_then(|_| unsigned[4..].strip_suffix(')'));
    if payload.is_some_and(|payload| {
        payload
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    }) {
        // The sign and `nan`, without the sequence.
        return s[..s.len() - unsigned.len() + 3].parse().ok();
    }
    s.parse().ok()
}

/// Formats the values of `array` as the strings of `to_type` as Arrow C++
/// `StringFormatter` and its zoned timestamp formatter do, or returns `None`
/// when Arrow C++ does not format the input type this way.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for a timestamp time zone that is not a
/// time zone.
pub(crate) fn conversion_format(
    array: &dyn Array,
    to_type: &DataType,
) -> Option<Result<ArrayRef, ArrowError>> {
    formatted(array, to_type).transpose()
}

fn formatted(array: &dyn Array, to_type: &DataType) -> Result<Option<ArrayRef>, ArrowError> {
    use DataType::*;
    use std::fmt::Write;
    // Only a string target formats, so no other target reads the time zone.
    if !matches!(to_type, Utf8 | LargeUtf8 | Utf8View) {
        return Ok(None);
    }
    type Format<'a> = Box<dyn Fn(usize, &mut String) + 'a>;
    fn integer<T: ArrowPrimitiveType>(array: &dyn Array) -> Format<'_>
    where
        T::Native: std::fmt::Display,
    {
        let array = array.as_primitive::<T>();
        Box::new(move |i, out| write!(out, "{}", array.value(i)).expect("writes to a String"))
    }
    fn decimal<T: ArrowPrimitiveType>(array: &dyn Array, scale: i8) -> Format<'_>
    where
        T::Native: std::fmt::Display,
    {
        let array = array.as_primitive::<T>();
        Box::new(move |i, out| {
            write!(out, "{}", array.value(i)).expect("writes to a String");
            adjust_integer_string_with_scale(scale as i32, out);
        })
    }
    let format: Format = match array.data_type() {
        Boolean => {
            let array = array.as_boolean();
            Box::new(move |i, out| out.push_str(if array.value(i) { "true" } else { "false" }))
        }
        Int8 => integer::<Int8Type>(array),
        Int16 => integer::<Int16Type>(array),
        Int32 => integer::<Int32Type>(array),
        Int64 => integer::<Int64Type>(array),
        UInt8 => integer::<UInt8Type>(array),
        UInt16 => integer::<UInt16Type>(array),
        UInt32 => integer::<UInt32Type>(array),
        UInt64 => integer::<UInt64Type>(array),
        Duration(_) => {
            let value = wide_values(array)?;
            Box::new(move |i, out| write!(out, "{}", value(i)).expect("writes to a String"))
        }
        Float16 => {
            let array = array.as_primitive::<Float16Type>();
            Box::new(move |i, out| {
                let shortest = format!("{:e}", array.value(i).to_f32() as f64);
                format_cpp_float(&shortest, out)
            })
        }
        Float32 => {
            let array = array.as_primitive::<Float32Type>();
            Box::new(move |i, out| format_cpp_float(&format!("{:e}", array.value(i)), out))
        }
        Float64 => {
            let array = array.as_primitive::<Float64Type>();
            Box::new(move |i, out| format_cpp_float(&format!("{:e}", array.value(i)), out))
        }
        Decimal32(_, scale) => decimal::<Decimal32Type>(array, *scale),
        Decimal64(_, scale) => decimal::<Decimal64Type>(array, *scale),
        Decimal128(_, scale) => decimal::<Decimal128Type>(array, *scale),
        Decimal256(_, scale) => decimal::<Decimal256Type>(array, *scale),
        Date32 => {
            let value = wide_values(array)?;
            Box::new(move |i, out| {
                let days = value(i);
                if in_date_range(days, 1) {
                    format_date(days, out)
                } else {
                    write!(out, "<value out of range: {days}>").expect("writes to a String")
                }
            })
        }
        Date64 => {
            let value = wide_values(array)?;
            Box::new(move |i, out| {
                let millis = value(i);
                if in_date_range(millis, 86_400_000) {
                    // Arrow C++ truncates milliseconds to days towards zero.
                    format_date(millis / 86_400_000, out)
                } else {
                    write!(out, "<value out of range: {millis}>").expect("writes to a String")
                }
            })
        }
        Time32(unit) | Time64(unit) => {
            let per_second = per_second(unit);
            let value = wide_values(array)?;
            Box::new(move |i, out| {
                let value = value(i);
                if (0..86_400 * per_second).contains(&value) {
                    format_time(value, per_second, out)
                } else {
                    write!(out, "<value out of range: {value}>").expect("writes to a String")
                }
            })
        }
        Timestamp(unit, tz) => {
            let zone = zone(tz.as_deref())?;
            let utc = tz.as_deref() == Some("UTC");
            let per_second = per_second(unit);
            let per_day = per_second * 86_400;
            let value = wide_values(array)?;
            let nanoseconds = matches!(unit, TimeUnit::Nanosecond);
            Box::new(move |i, out| {
                let value = value(i);
                if !nanoseconds && !in_date_range(value, per_day) {
                    write!(out, "<value out of range: {value}>").expect("writes to a String");
                    return;
                }
                let offset = zone
                    .as_ref()
                    .map_or(0, |zone| utc_offset(zone, value, per_second));
                let local = value.wrapping_add(offset as i64 * per_second);
                format_date(local.div_euclid(per_day), out);
                out.push(' ');
                format_time(local.rem_euclid(per_day), per_second, out);
                if utc {
                    out.push('Z');
                } else if zone.is_some() {
                    let sign = if offset < 0 { '-' } else { '+' };
                    let minutes = offset.abs() / 60;
                    write!(out, "{sign}{:02}{:02}", minutes / 60, minutes % 60)
                        .expect("writes to a String");
                }
            })
        }
        _ => return Ok(None),
    };
    let mut buffer = String::new();
    let mut value = |i: usize| -> Option<String> {
        array.is_valid(i).then(|| {
            buffer.clear();
            format(i, &mut buffer);
            buffer.clone()
        })
    };
    Ok(Some(match to_type {
        Utf8 => Arc::new((0..array.len()).map(&mut value).collect::<StringArray>()),
        LargeUtf8 => Arc::new(
            (0..array.len())
                .map(&mut value)
                .collect::<LargeStringArray>(),
        ),
        Utf8View => Arc::new(
            (0..array.len())
                .map(&mut value)
                .collect::<StringViewArray>(),
        ),
        _ => return Ok(None),
    }))
}

/// The values of a temporal `array`, sign extended to `i64`.
fn wide_values(array: &dyn Array) -> Result<Box<dyn Fn(usize) -> i64>, ArrowError> {
    Ok(if array.data_type().primitive_width() == Some(4) {
        let values = retype(array, DataType::Int32)?
            .as_primitive::<Int32Type>()
            .clone();
        Box::new(move |i| values.value(i) as i64)
    } else {
        let values = retype(array, DataType::Int64)?
            .as_primitive::<Int64Type>()
            .clone();
        Box::new(move |i| values.value(i))
    })
}

/// Whether the instant `value`, counted in units of which `per_day` make a
/// day, is within the years -32767 to 32767 Arrow C++ formats.
fn in_date_range(value: i64, per_day: i64) -> bool {
    (-12_687_428 * per_day as i128..11_248_738 * per_day as i128).contains(&(value as i128))
}

/// Appends the proleptic Gregorian date `days` after 1970-01-01 as Arrow C++
/// `FormatYYYY_MM_DD` does: at least four year digits and a leading `-` for
/// years before year zero.
fn format_date(days: i64, out: &mut String) {
    use chrono::Datelike;
    use std::fmt::Write;
    let date = chrono::NaiveDate::from_num_days_from_ce_opt((days + 719_163) as i32)
        .expect("a date in the formatted range");
    let year = date.year();
    let sign = if year < 0 { "-" } else { "" };
    write!(
        out,
        "{sign}{:04}-{:02}-{:02}",
        year.abs(),
        date.month(),
        date.day()
    )
    .expect("writes to a String");
}

/// Appends the time of day `value`, counted in units of which `per_second`
/// make a second, with one fractional digit per decimal digit of the unit.
fn format_time(value: i64, per_second: i64, out: &mut String) {
    use std::fmt::Write;
    let seconds = value / per_second;
    write!(
        out,
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
    .expect("writes to a String");
    if per_second > 1 {
        let digits = per_second.ilog10() as usize;
        write!(out, ".{:0digits$}", value % per_second).expect("writes to a String");
    }
}

/// Appends the shortest round trip representation `scientific`, written by
/// Rust as `{:e}`, as Arrow C++ formats floating point values: decimal
/// notation for decimal exponents from -6 to 9, scientific notation with a
/// signed exponent otherwise, and `inf`, `-inf` and `nan`.
fn format_cpp_float(scientific: &str, out: &mut String) {
    use std::fmt::Write;
    if scientific.ends_with("inf") || scientific == "NaN" {
        out.push_str(match scientific {
            "NaN" => "nan",
            "inf" => "inf",
            _ => "-inf",
        });
        return;
    }
    let (mantissa, exponent) = scientific.split_once('e').expect("Rust writes an exponent");
    let exponent: i32 = exponent.parse().expect("Rust writes an integer exponent");
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(mantissa) => (true, mantissa),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let length = digits.len() as i32;
    if negative {
        out.push('-');
    }
    if (-6..10).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point >= length {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (point - length) as usize));
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if length > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        write!(out, "e{sign}{}", exponent.abs()).expect("writes to a String");
    }
}

/// Arrow C++ `AdjustIntegerStringWithScale`: places the decimal point of the
/// integer string `out` for `scale`, in scientific notation for a negative
/// scale or an adjusted exponent below -6.
fn adjust_integer_string_with_scale(scale: i32, out: &mut String) {
    use std::fmt::Write;
    if scale == 0 {
        return;
    }
    let negative = out.starts_with('-') as usize;
    let length = out.len() as i32;
    let digits = length - negative as i32;
    let adjusted_exponent = digits - 1 - scale;
    if scale < 0 || adjusted_exponent < -6 {
        if digits > 1 {
            out.insert(1 + negative, '.');
        }
        out.push('E');
        if adjusted_exponent >= 0 {
            out.push('+');
        }
        write!(out, "{adjusted_exponent}").expect("writes to a String");
    } else if digits > scale {
        out.insert((length - scale) as usize, '.');
    } else {
        let zeros: String = std::iter::repeat_n('0', (scale - digits + 2) as usize).collect();
        out.insert_str(negative, &zeros);
        out.replace_range(negative + 1..negative + 2, ".");
    }
}

/// The value of the two to four ASCII digits `s`, as Arrow C++ `ParseUnsigned`
/// reads a fixed width field.
fn fixed_digits(s: &[u8]) -> Option<u32> {
    s.iter().try_fold(0u32, |value, digit| {
        digit
            .is_ascii_digit()
            .then(|| value * 10 + (digit - b'0') as u32)
    })
}

/// Arrow C++ `ParseYYYY_MM_DD` of the first ten bytes of `s`: days since
/// 1970-01-01 of a valid proleptic Gregorian date.
fn parse_cpp_date(s: &[u8]) -> Option<i64> {
    use chrono::Datelike;
    if s.len() < 10 || s[4] != b'-' || s[7] != b'-' {
        return None;
    }
    let date = chrono::NaiveDate::from_ymd_opt(
        fixed_digits(&s[0..4])? as i32,
        fixed_digits(&s[5..7])?,
        fixed_digits(&s[8..10])?,
    )?;
    Some(date.num_days_from_ce() as i64 - 719_163)
}

/// Arrow C++ `ParseHH`, `ParseHH_MM`, `ParseHHMM` and `ParseHH_MM_SS`: the
/// seconds of an `hh`, `hh:mm`, `hhmm` or `hh:mm:ss` field.
fn parse_cpp_clock(s: &[u8]) -> Option<i64> {
    let field = |range: std::ops::Range<usize>, limit| {
        fixed_digits(&s[range]).filter(|value| *value < limit)
    };
    let (hours, minutes, seconds) = match s.len() {
        2 => (field(0..2, 24)?, 0, 0),
        4 => (field(0..2, 24)?, field(2..4, 60)?, 0),
        5 if s[2] == b':' => (field(0..2, 24)?, field(3..5, 60)?, 0),
        8 if s[2] == b':' && s[5] == b':' => (field(0..2, 24)?, field(3..5, 60)?, field(6..8, 60)?),
        _ => return None,
    };
    Some(hours as i64 * 3600 + minutes as i64 * 60 + seconds as i64)
}

/// Arrow C++ `ParseTimestampISO8601`: the instant in `unit` of a
/// `YYYY-MM-DD[( |T)hh[:mm[:ss[.s{1,9}]]]][Z|(+|-)hh[[:]mm]]` string, and
/// whether it carries a zone offset. Fractional seconds may have at most the
/// digits of `unit`.
fn parse_cpp_timestamp(s: &[u8], unit: &TimeUnit) -> Option<(i64, bool)> {
    let per_second = per_second(unit);
    let date = parse_cpp_date(s)?;
    if s.len() == 10 {
        return Some(((date * 86_400).checked_mul(per_second)?, false));
    }
    if s[10] != b' ' && s[10] != b'T' {
        return None;
    }
    let mut length = s.len();
    let mut offset = 0;
    let mut zoned = false;
    let sign_at =
        |position: usize| length >= position && matches!(s[length - position], b'+' | b'-');
    if s[length - 1] == b'Z' {
        length -= 1;
        zoned = true;
    } else if sign_at(3) || sign_at(5) || (sign_at(6) && s[length - 3] == b':') {
        let width = if sign_at(3) {
            3
        } else if sign_at(5) {
            5
        } else {
            6
        };
        length -= width;
        offset = parse_cpp_clock(&s[length + 1..length + width])?;
        if s[length] == b'+' {
            offset = -offset;
        }
        zoned = true;
    }
    let clock = match length {
        13 | 16 | 19 => parse_cpp_clock(&s[11..length])?,
        21..=29 => parse_cpp_clock(&s[11..19])?,
        _ => return None,
    };
    let seconds = (clock + offset).checked_add(date * 86_400)?;
    let mut value = seconds.checked_mul(per_second)?;
    if length > 19 {
        let fraction = &s[20..length];
        let digits = per_second.ilog10() as usize;
        if s[19] != b'.' || fraction.len() > digits {
            return None;
        }
        let subseconds =
            fixed_digits(fraction)? as i64 * 10_i64.pow((digits - fraction.len()) as u32);
        value = value.checked_add(subseconds)?;
    }
    Some((value, zoned))
}

/// Parses strings to timestamps of `unit` as Arrow C++ does: ISO 8601 with a
/// zone offset exactly when the target has a time zone.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first string that does not parse
/// or whose zone offset presence does not match the target.
pub(crate) fn conversion_parse_timestamp(
    array: &dyn Array,
    to_type: &DataType,
) -> Result<ArrayRef, ArrowError> {
    let DataType::Timestamp(unit, tz) = to_type else {
        unreachable!("only called for timestamp targets")
    };
    let expect_zone = tz.is_some();
    let values = byte_values(array)
        .map(|value| {
            value
                .map(|s| {
                    let error = |detail: &str| {
                        ArrowError::CastError(format!(
                            "Failed to parse string: '{}' as a scalar of type {to_type}{detail}",
                            String::from_utf8_lossy(s)
                        ))
                    };
                    let (value, zoned) = parse_cpp_timestamp(s, unit).ok_or_else(|| error(""))?;
                    match (zoned, expect_zone) {
                        (false, true) => Err(error(": expected a zone offset")),
                        (true, false) => Err(error(": expected no zone offset")),
                        _ => Ok(value),
                    }
                })
                .transpose()
        })
        .collect::<Result<PrimitiveArray<Int64Type>, _>>()?;
    retype(&values, to_type.clone())
}

/// Parses `YYYY-MM-DD` strings to dates as Arrow C++ does.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first string that is not exactly
/// a valid date.
pub(crate) fn conversion_parse_date(
    array: &dyn Array,
    to_type: &DataType,
) -> Result<ArrayRef, ArrowError> {
    let scale = if matches!(to_type, DataType::Date64) {
        86_400_000
    } else {
        1
    };
    let values = byte_values(array)
        .map(|value| {
            value
                .map(|s| {
                    (s.len() == 10)
                        .then(|| parse_cpp_date(s))
                        .flatten()
                        .map(|days| days * scale)
                        .ok_or_else(|| {
                            ArrowError::CastError(format!(
                                "Failed to parse string: '{}' as a scalar of type {to_type}",
                                String::from_utf8_lossy(s)
                            ))
                        })
                })
                .transpose()
        })
        .collect::<Result<PrimitiveArray<Int64Type>, _>>()?;
    if scale == 1 {
        retype(&values.unary::<_, Int32Type>(|v| v as i32), to_type.clone())
    } else {
        retype(&values, to_type.clone())
    }
}
