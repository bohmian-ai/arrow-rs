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

//! Conversions between temporal types as Arrow C++ performs them.

use crate::cast::*;
use chrono::DateTime;

/// Units of `unit` in one second.
pub(crate) fn per_second(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

/// Arrow C++ `GetTimestampConversion`: whether converting `from` to `to`
/// multiplies, and by which factor.
fn unit_conversion(from: &TimeUnit, to: &TimeUnit) -> (bool, i64) {
    let (from, to) = (per_second(from), per_second(to));
    if to >= from {
        (true, to / from)
    } else {
        (false, from / to)
    }
}

/// The unit of a time type.
fn time_unit(data_type: &DataType) -> &TimeUnit {
    match data_type {
        DataType::Time32(unit) | DataType::Time64(unit) => unit,
        _ => unreachable!("only called for time types"),
    }
}

/// `array` with its values read as `data_type`, sharing its buffers.
pub(crate) fn retype(array: &dyn Array, data_type: DataType) -> Result<ArrayRef, ArrowError> {
    Ok(make_array(
        array
            .to_data()
            .into_builder()
            .data_type(data_type)
            .build()?,
    ))
}

/// Converts temporal values between the types of `from` and `to` as Arrow C++
/// does, or returns `None` when the pair is not a temporal conversion: units
/// and dates shift with Arrow C++ `ShiftTime`, and timestamps convert to dates
/// and times of day in their time zone.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for an unknown time zone, a value that
/// overflows without `allow_time_overflow`, or one that loses precision
/// without `allow_time_truncate`.
pub(crate) fn conversion_temporal(
    array: &dyn Array,
    to_type: &DataType,
    conversion: &ConversionOptions,
) -> Option<Result<ArrayRef, ArrowError>> {
    use DataType::*;
    const MILLIS_PER_DAY: i64 = 86_400_000;
    let shift = |multiply, factor| {
        Some(match (array.data_type(), to_type) {
            (Date32 | Time32(_), Date64 | Time64(_) | Timestamp(..)) => {
                shift_time::<Int32Type, Int64Type>(array, to_type, multiply, factor, conversion)
            }
            (Date64 | Time64(_) | Timestamp(..), Date32 | Time32(_)) => {
                shift_time::<Int64Type, Int32Type>(array, to_type, multiply, factor, conversion)
            }
            (Date32 | Time32(_), _) => {
                shift_time::<Int32Type, Int32Type>(array, to_type, multiply, factor, conversion)
            }
            _ => shift_time::<Int64Type, Int64Type>(array, to_type, multiply, factor, conversion),
        })
    };
    match (array.data_type(), to_type) {
        (Timestamp(from, _), Timestamp(to, _)) | (Duration(from), Duration(to)) => {
            if from == to {
                Some(retype(array, to_type.clone()))
            } else {
                let (multiply, factor) = unit_conversion(from, to);
                shift(multiply, factor)
            }
        }
        (Time32(_) | Time64(_), Time32(_) | Time64(_)) => {
            let (multiply, factor) =
                unit_conversion(time_unit(array.data_type()), time_unit(to_type));
            shift(multiply, factor)
        }
        (Date32, Date64) => shift(true, MILLIS_PER_DAY),
        (Date64, Date32) => shift(false, MILLIS_PER_DAY),
        (Date32, Timestamp(to, _)) => shift(true, per_second(to) * 86_400),
        (Date64, Timestamp(to, _)) => {
            let (multiply, factor) = unit_conversion(&TimeUnit::Millisecond, to);
            shift(multiply, factor)
        }
        (Timestamp(unit, tz), Date32 | Date64 | Time32(_) | Time64(_)) => Some(
            localized_timestamp(array, unit, tz.as_deref(), to_type, conversion),
        ),
        _ => None,
    }
}

/// Arrow C++ `ShiftTime`: multiplies or divides every value by `factor`,
/// refusing a product outside the `i64` range unless `allow_time_overflow`
/// and a quotient with a remainder unless `allow_time_truncate`. The result
/// keeps the low bits of the value.
fn shift_time<I, O>(
    array: &dyn Array,
    to_type: &DataType,
    multiply: bool,
    factor: i64,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError>
where
    I: ArrowPrimitiveType,
    O: ArrowPrimitiveType,
    I::Native: AsPrimitive<i64>,
    i64: AsPrimitive<O::Native>,
    O::Native: AsPrimitive<i64>,
{
    let values = retype(array, I::DATA_TYPE)?;
    let values = values.as_primitive::<I>();
    let from_type = array.data_type();
    let shifted: PrimitiveArray<O> = if factor == 1 {
        values.unary(|v| v.as_().as_())
    } else if multiply {
        if conversion.allow_time_overflow {
            values.unary(|v| v.as_().wrapping_mul(factor).as_())
        } else {
            let (min, max) = (i64::MIN / factor, i64::MAX / factor);
            values.try_unary(|v| {
                let v: i64 = v.as_();
                if v < min || v > max {
                    return Err(ArrowError::CastError(format!(
                        "Casting from {from_type} to {to_type} would result in out of bounds timestamp: {v}"
                    )));
                }
                Ok((v * factor).as_())
            })?
        }
    } else if conversion.allow_time_truncate {
        values.unary(|v| (v.as_() / factor).as_())
    } else {
        values.try_unary(|v| {
            let v: i64 = v.as_();
            let shifted: O::Native = (v / factor).as_();
            if shifted.as_().wrapping_mul(factor) != v {
                return Err(ArrowError::CastError(format!(
                    "Casting from {from_type} to {to_type} would lose data: {v}"
                )));
            }
            Ok(shifted)
        })?
    };
    retype(&shifted, to_type.clone())
}

/// Arrow C++ timestamp to date and time of day conversion: each instant is
/// moved to the local time of `tz`, then floored to its day, or reduced to the
/// time since its local midnight in the target unit.
fn localized_timestamp(
    array: &dyn Array,
    unit: &TimeUnit,
    tz: Option<&str>,
    to_type: &DataType,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError> {
    let tz = zone(tz)?;
    let per_second = per_second(unit);
    let per_day = per_second * 86_400;
    let local = move |v: i64| match &tz {
        None => v,
        Some(tz) => v.wrapping_add(utc_offset(tz, v, per_second) as i64 * per_second),
    };
    let values = retype(array, DataType::Int64)?;
    let values = values.as_primitive::<Int64Type>();
    let days = |v: i64| local(v).div_euclid(per_day) as i32;
    let time_of_day = |v: i64| local(v).rem_euclid(per_day);
    let result: ArrayRef = match to_type {
        DataType::Date32 => Arc::new(values.unary::<_, Int32Type>(days)),
        DataType::Date64 => Arc::new(values.unary::<_, Int64Type>(|v| days(v) as i64 * 86_400_000)),
        DataType::Time32(to) | DataType::Time64(to) => {
            let (multiply, factor) = unit_conversion(unit, to);
            let wide = matches!(to_type, DataType::Time64(_));
            let scaled: PrimitiveArray<Int64Type> = if multiply {
                values.unary(|v| time_of_day(v).wrapping_mul(factor))
            } else if conversion.allow_time_truncate {
                values.unary(|v| time_of_day(v) / factor)
            } else {
                values.try_unary(|v| {
                    let time = time_of_day(v);
                    if time % factor != 0 {
                        return Err(ArrowError::CastError(format!(
                            "Cast would lose data: {time}"
                        )));
                    }
                    Ok(time / factor)
                })?
            };
            if wide {
                Arc::new(scaled)
            } else {
                Arc::new(scaled.unary::<_, Int32Type>(|v| v as i32))
            }
        }
        _ => unreachable!("only called for date and time targets"),
    };
    retype(&result, to_type.clone())
}

/// The time zone named by a timestamp type, as Arrow C++ `LocateZone` finds it.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for a name that is not a time zone.
pub(crate) fn zone(tz: Option<&str>) -> Result<Option<Tz>, ArrowError> {
    // Arrow C++ reads an empty time zone as none.
    tz.filter(|tz| !tz.is_empty()).map(|tz| {
        let refused = |error: String| {
            ArrowError::CastError(format!("Cannot locate or parse timezone '{tz}': {error}"))
        };
        match tz.strip_prefix(['+', '-']) {
            Some(offset) => {
                let (hours, minutes) =
                    cpp_offset(tz.len(), offset).ok_or_else(|| refused("not an offset".into()))?;
                format!("{}{hours:02}:{minutes:02}", &tz[..1]).parse::<Tz>()
            }
            None => tz.parse::<Tz>(),
        }
        .map_err(|error| refused(error.to_string()))
    })
    .transpose()
}

/// The hours and minutes of the offset of the zone `+offset` or `-offset`,
/// `len` characters long, as Arrow C++ `LocateZone` reads them: `HH:MM` in
/// six characters, else the first four characters as `HHMM` in five or six,
/// with fewer than 24 hours and 60 minutes.
fn cpp_offset(len: usize, offset: &str) -> Option<(u8, u8)> {
    let number = |range: std::ops::Range<usize>| {
        let digits = offset.get(range)?;
        digits
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| digits.parse::<u8>().ok())?
    };
    let colon = len == 6 && offset.as_bytes().get(2) == Some(&b':');
    let parsed = match (colon, number(0..2), number(3..5)) {
        (true, Some(hours), Some(minutes)) if hours < 24 && minutes < 60 => Some((hours, minutes)),
        _ => None,
    };
    parsed.or_else(|| match (len, number(0..2), number(2..4)) {
        (5 | 6, Some(hours), Some(minutes)) if hours < 24 && minutes < 60 => Some((hours, minutes)),
        _ => None,
    })
}

/// The offset from UTC in seconds of `tz` at the instant `value`, counted in
/// units of which `per_second` make a second.
pub(crate) fn utc_offset(tz: &Tz, value: i64, per_second: i64) -> i32 {
    DateTime::from_timestamp(value.div_euclid(per_second), 0).map_or(0, |utc| {
        tz.offset_from_utc_datetime(&utc.naive_utc())
            .fix()
            .local_minus_utc()
    })
}
