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
use crate::parse::{DecimalParseError, parse_decimal_checked};

/// A utility trait that provides checked conversions between
/// decimal types inspired by [`NumCast`]
pub trait DecimalCast: Sized {
    /// Convert the decimal to an i32
    fn to_i32(self) -> Option<i32>;

    /// Convert the decimal to an i64
    fn to_i64(self) -> Option<i64>;

    /// Convert the decimal to an i128
    fn to_i128(self) -> Option<i128>;

    /// Convert the decimal to an i256
    fn to_i256(self) -> Option<i256>;

    /// Convert a decimal from a decimal
    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self>;

    /// Convert a decimal from a f64
    fn from_f64(n: f64) -> Option<Self>;
}

impl DecimalCast for i32 {
    fn to_i32(self) -> Option<i32> {
        Some(self)
    }

    fn to_i64(self) -> Option<i64> {
        Some(self as i64)
    }

    fn to_i128(self) -> Option<i128> {
        Some(self as i128)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self as i128))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i32()
    }

    fn from_f64(n: f64) -> Option<Self> {
        n.to_i32()
    }
}

impl DecimalCast for i64 {
    fn to_i32(self) -> Option<i32> {
        i32::try_from(self).ok()
    }

    fn to_i64(self) -> Option<i64> {
        Some(self)
    }

    fn to_i128(self) -> Option<i128> {
        Some(self as i128)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self as i128))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i64()
    }

    fn from_f64(n: f64) -> Option<Self> {
        // Call implementation explicitly otherwise this resolves to `to_i64`
        // in arrow-buffer that behaves differently.
        num_traits::ToPrimitive::to_i64(&n)
    }
}

impl DecimalCast for i128 {
    fn to_i32(self) -> Option<i32> {
        i32::try_from(self).ok()
    }

    fn to_i64(self) -> Option<i64> {
        i64::try_from(self).ok()
    }

    fn to_i128(self) -> Option<i128> {
        Some(self)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i128()
    }

    fn from_f64(n: f64) -> Option<Self> {
        n.to_i128()
    }
}

impl DecimalCast for i256 {
    fn to_i32(self) -> Option<i32> {
        self.to_i128().map(|x| i32::try_from(x).ok())?
    }

    fn to_i64(self) -> Option<i64> {
        self.to_i128().map(|x| i64::try_from(x).ok())?
    }

    fn to_i128(self) -> Option<i128> {
        self.to_i128()
    }

    fn to_i256(self) -> Option<i256> {
        Some(self)
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i256()
    }

    fn from_f64(n: f64) -> Option<Self> {
        i256::from_f64(n)
    }
}

/// Construct closures to upscale decimals from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)`.
///
/// Returns `(f_fallible, f_infallible)` where:
/// * `f_fallible` yields `None` when the requested cast would overflow
/// * `f_infallible` is present only when every input is guaranteed to succeed; otherwise it is `None`
///   and callers must fall back to `f_fallible`
///
/// Returns `None` if the required scale increase `delta_scale = output_scale - input_scale`
/// exceeds the supported precomputed precision table `O::MAX_FOR_EACH_PRECISION`.
/// In that case, the caller should treat this as an overflow for the output scale
/// and handle it accordingly (e.g., return a cast error).
#[expect(clippy::type_complexity)]
fn make_upscaler<I: DecimalType, O: DecimalType>(
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
) -> Option<(
    impl Fn(I::Native) -> Option<O::Native>,
    Option<impl Fn(I::Native) -> O::Native>,
)>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let delta_scale = output_scale - input_scale;

    // O::MAX_FOR_EACH_PRECISION[k] stores 10^k - 1 (e.g., 9, 99, 999, ...).
    // Adding 1 yields exactly 10^k without computing a power at runtime.
    // Using the precomputed table avoids pow(10, k) and its checked/overflow
    // handling, which is faster and simpler for scaling by 10^delta_scale.
    let max = O::MAX_FOR_EACH_PRECISION.get(delta_scale as usize)?;
    let mul = max.add_wrapping(O::Native::ONE);
    let f_fallible = move |x| O::Native::from_decimal(x)?.mul_checked(mul).ok();

    // if the gain in precision (digits) is greater than the multiplication due to scaling
    // every number will fit into the output type
    // Example: If we are starting with any number of precision 5 [xxxxx],
    // then an increase of scale by 3 will have the following effect on the representation:
    // [xxxxx] -> [xxxxx000], so for the cast to be infallible, the output type
    // needs to provide at least 8 digits precision
    let is_infallible_cast = (input_precision as i8) + delta_scale <= (output_precision as i8);
    let f_infallible = is_infallible_cast
        .then_some(move |x| O::Native::from_decimal(x).unwrap().mul_wrapping(mul));
    Some((f_fallible, f_infallible))
}

/// Construct closures to downscale decimals from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)`.
///
/// Returns `(f_fallible, f_infallible)` where:
/// * `f_fallible` yields `None` when the requested cast would overflow
/// * `f_infallible` is present only when every input is guaranteed to succeed; otherwise it is `None`
///   and callers must fall back to `f_fallible`
///
/// Returns `None` if the required scale reduction `delta_scale = input_scale - output_scale`
/// exceeds the supported precomputed precision table `I::MAX_FOR_EACH_PRECISION`.
/// In this scenario, any value would round to zero (e.g., dividing by 10^k where k exceeds the
/// available precision). Callers should therefore produce zero values (preserving nulls) rather
/// than returning an error.
#[expect(clippy::type_complexity)]
fn make_downscaler<I: DecimalType, O: DecimalType>(
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
) -> Option<(
    impl Fn(I::Native) -> Option<O::Native>,
    Option<impl Fn(I::Native) -> O::Native>,
)>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let delta_scale = input_scale - output_scale;

    // delta_scale is guaranteed to be > 0, but may also be larger than I::MAX_PRECISION. If so, the
    // scale change divides out more digits than the input has precision and the result of the cast
    // is always zero. For example, if we try to apply delta_scale=10 a decimal32 value, the largest
    // possible result is 999999999/10000000000 = 0.0999999999, which rounds to zero. Smaller values
    // (e.g. 1/10000000000) or larger delta_scale (e.g. 999999999/10000000000000) produce even
    // smaller results, which also round to zero. In that case, just return an array of zeros.
    let max = I::MAX_FOR_EACH_PRECISION.get(delta_scale as usize)?;

    let div = max.add_wrapping(I::Native::ONE);
    let half = div.div_wrapping(I::Native::ONE.add_wrapping(I::Native::ONE));
    let half_neg = half.neg_wrapping();

    let f_fallible = move |x: I::Native| {
        // div is >= 10 and so this cannot overflow
        let d = x.div_wrapping(div);
        let r = x.mod_wrapping(div);

        // Round result
        let adjusted = match x >= I::Native::ZERO {
            true if r >= half => d.add_wrapping(I::Native::ONE),
            false if r <= half_neg => d.sub_wrapping(I::Native::ONE),
            _ => d,
        };
        O::Native::from_decimal(adjusted)
    };

    // if the reduction of the input number through scaling (dividing) is greater
    // than a possible precision loss (plus potential increase via rounding)
    // every input number will fit into the output type
    // Example: If we are starting with any number of precision 5 [xxxxx],
    // then and decrease the scale by 3 will have the following effect on the representation:
    // [xxxxx] -> [xx] (+ 1 possibly, due to rounding).
    // The rounding may add a digit, so the cast to be infallible,
    // the output type needs to have at least 3 digits of precision.
    // e.g. Decimal(5, 3) 99.999 to Decimal(3, 0) will result in 100:
    // [99999] -> [99] + 1 = [100], a cast to Decimal(2, 0) would not be possible
    let is_infallible_cast = (input_precision as i8) - delta_scale < (output_precision as i8);
    let f_infallible = is_infallible_cast.then_some(move |x| f_fallible(x).unwrap());
    Some((f_fallible, f_infallible))
}

/// Apply the rescaler function to the value.
/// If the rescaler is infallible, use the infallible function.
/// Otherwise, use the fallible function and validate the precision.
fn apply_rescaler<I: DecimalType, O: DecimalType>(
    value: I::Native,
    output_precision: u8,
    f: impl Fn(I::Native) -> Option<O::Native>,
    f_infallible: Option<impl Fn(I::Native) -> O::Native>,
) -> Option<O::Native>
where
    I::Native: DecimalCast,
    O::Native: DecimalCast,
{
    if let Some(f_infallible) = f_infallible {
        Some(f_infallible(value))
    } else {
        f(value).filter(|v| O::is_valid_decimal_precision(*v, output_precision))
    }
}

/// Rescales a decimal value from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)` and returns the converted number when it fits
/// within the output precision.
///
/// The function first validates that the requested precision and scale are supported for
/// both the source and destination decimal types. It then either upscales (multiplying
/// by an appropriate power of ten) or downscales (dividing with rounding) the input value.
/// When the scaling factor exceeds the precision table of the destination type, the value
/// is treated as an overflow for upscaling, or rounded to zero for downscaling (as any
/// possible result would be zero at the requested scale).
///
/// This mirrors the column-oriented helpers of decimal casting but operates on a single value
/// (row-level) instead of an entire array.
///
/// Returns `None` if the value cannot be represented with the requested precision.
pub fn rescale_decimal<I: DecimalType, O: DecimalType>(
    value: I::Native,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
) -> Option<O::Native>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    validate_decimal_precision_and_scale::<I>(input_precision, input_scale).ok()?;
    validate_decimal_precision_and_scale::<O>(output_precision, output_scale).ok()?;

    if input_scale <= output_scale {
        let (f, f_infallible) =
            make_upscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)?;
        apply_rescaler::<I, O>(value, output_precision, f, f_infallible)
    } else {
        let Some((f, f_infallible)) =
            make_downscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)
        else {
            // Scale reduction exceeds supported precision; result mathematically rounds to zero
            return Some(O::Native::ZERO);
        };
        apply_rescaler::<I, O>(value, output_precision, f, f_infallible)
    }
}

fn cast_decimal_to_decimal_error<I, O>(
    output_precision: u8,
    output_scale: i8,
) -> impl Fn(<I as ArrowPrimitiveType>::Native) -> ArrowError
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    move |x: I::Native| {
        ArrowError::CastError(format!(
            "Cannot cast to {}({}, {}). Overflowing on {:?}",
            O::PREFIX,
            output_precision,
            output_scale,
            x
        ))
    }
}

fn apply_decimal_cast<I: DecimalType, O: DecimalType>(
    array: &PrimitiveArray<I>,
    output_precision: u8,
    output_scale: i8,
    f_fallible: impl Fn(I::Native) -> Option<O::Native>,
    f_infallible: Option<impl Fn(I::Native) -> O::Native>,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array = if let Some(f_infallible) = f_infallible {
        array.unary(f_infallible)
    } else if cast_options.safe {
        array.unary_opt(|x| {
            f_fallible(x).filter(|v| O::is_valid_decimal_precision(*v, output_precision))
        })
    } else {
        let error = cast_decimal_to_decimal_error::<I, O>(output_precision, output_scale);
        array.try_unary(|x| {
            let v = f_fallible(x).ok_or_else(|| error(x))?;
            O::validate_decimal_precision(v, output_precision, output_scale).map(|()| v)
        })?
    };
    Ok(array)
}

fn convert_to_smaller_scale_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    if let Some((f_fallible, f_infallible)) =
        make_downscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)
    {
        apply_decimal_cast(
            array,
            output_precision,
            output_scale,
            f_fallible,
            f_infallible,
            cast_options,
        )
    } else {
        // Scale reduction exceeds supported precision; result mathematically rounds to zero
        let zeros = vec![O::Native::ZERO; array.len()];
        Ok(PrimitiveArray::new(zeros.into(), array.nulls().cloned()))
    }
}

fn convert_to_bigger_or_equal_scale_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    if let Some((f, f_infallible)) =
        make_upscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)
    {
        apply_decimal_cast(
            array,
            output_precision,
            output_scale,
            f,
            f_infallible,
            cast_options,
        )
    } else {
        // Scale increase exceeds supported precision; return overflow error
        Err(ArrowError::CastError(format!(
            "Cannot cast to {}({}, {}). Value overflows for output scale",
            O::PREFIX,
            output_precision,
            output_scale
        )))
    }
}

// Only support one type of decimal cast operations
pub(crate) fn cast_decimal_to_decimal_same_type<T>(
    array: &PrimitiveArray<T>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    T: DecimalType,
    T::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array: PrimitiveArray<T> =
        if input_scale == output_scale && input_precision <= output_precision {
            array.clone()
        } else if input_scale <= output_scale {
            convert_to_bigger_or_equal_scale_decimal::<T, T>(
                array,
                input_precision,
                input_scale,
                output_precision,
                output_scale,
                cast_options,
            )?
        } else {
            // input_scale > output_scale
            convert_to_smaller_scale_decimal::<T, T>(
                array,
                input_precision,
                input_scale,
                output_precision,
                output_scale,
                cast_options,
            )?
        };

    Ok(Arc::new(array.with_precision_and_scale(
        output_precision,
        output_scale,
    )?))
}

// Support two different types of decimal cast operations
pub(crate) fn cast_decimal_to_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array: PrimitiveArray<O> = if input_scale > output_scale {
        convert_to_smaller_scale_decimal::<I, O>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
        )?
    } else {
        convert_to_bigger_or_equal_scale_decimal::<I, O>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
        )?
    };

    Ok(Arc::new(array.with_precision_and_scale(
        output_precision,
        output_scale,
    )?))
}

/// Parses the given string as a decimal with the given scale, returning the
/// unscaled representation in the decimal type's native integer (e.g. `i32`
/// for `Decimal32Type`, `i256` for `Decimal256Type`).
///
/// Returns an error if the input is not a valid decimal string, or if the
/// scaled and rounded value does not fit the maximum precision of the decimal
/// type. The caller is responsible for validating the result against any
/// smaller target precision.
#[deprecated(
    since = "60.0.0",
    note = "Use `arrow_cast::parse::parse_decimal` instead"
)]
pub fn parse_string_to_decimal_native<T: DecimalType>(
    value_str: &str,
    scale: usize,
) -> Result<T::Native, ArrowError> {
    let overflow = || {
        ArrowError::InvalidArgumentError(format!(
            "Cannot convert {value_str} to {}: Overflow",
            T::PREFIX
        ))
    };
    let scale = i8::try_from(scale).map_err(|_| overflow())?;
    parse_decimal_checked::<T>(value_str, T::MAX_PRECISION, scale).map_err(|e| match e {
        DecimalParseError::InvalidFormat => {
            ArrowError::InvalidArgumentError(format!("Invalid decimal format: {value_str:?}"))
        }
        DecimalParseError::Overflow => overflow(),
    })
}

pub(crate) fn generic_string_to_decimal_cast<'a, T, S>(
    from: &'a S,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError>
where
    T: DecimalType,
    &'a S: StringArrayType<'a>,
{
    if cast_options.safe {
        let iter = from
            .iter()
            .map(|v| parse_decimal_checked::<T>(v?, precision, scale).ok());
        // Benefit:
        //     15-19% faster than appending to a PrimitiveBuilder (measured
        //     with the cast_kernels string-to-decimal benchmarks)
        // Soundness:
        //     The iterator is trustedLen because it comes from a `StringArray`.
        Ok(unsafe {
            PrimitiveArray::<T>::from_trusted_len_iter(iter)
                .with_precision_and_scale(precision, scale)?
        })
    } else {
        let mut builder = PrimitiveBuilder::<T>::with_capacity(from.len());
        for v in from.iter() {
            match v {
                Some(v) => {
                    let v = parse_decimal_checked::<T>(v, precision, scale).map_err(|e| {
                        let reason = match e {
                            DecimalParseError::InvalidFormat => "invalid decimal format",
                            DecimalParseError::Overflow => "value does not fit",
                        };
                        ArrowError::CastError(format!(
                            "Cannot cast string '{v}' to value of {}({precision}, {scale}) type: {reason}",
                            T::PREFIX,
                        ))
                    })?;
                    builder.append_value(v);
                }
                None => builder.append_null(),
            }
        }
        builder.finish().with_precision_and_scale(precision, scale)
    }
}

pub(crate) fn string_to_decimal_cast<T: DecimalType, Offset: OffsetSizeTrait>(
    from: &GenericStringArray<Offset>,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError> {
    generic_string_to_decimal_cast::<T, GenericStringArray<Offset>>(
        from,
        precision,
        scale,
        cast_options,
    )
}

pub(crate) fn string_view_to_decimal_cast<T: DecimalType>(
    from: &StringViewArray,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError> {
    generic_string_to_decimal_cast::<T, StringViewArray>(from, precision, scale, cast_options)
}

/// Cast Utf8 to decimal
pub(crate) fn cast_string_to_decimal<T: DecimalType, Offset: OffsetSizeTrait>(
    from: &dyn Array,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    validate_decimal_precision_and_scale::<T>(precision, scale)?;

    let result = match from.data_type() {
        DataType::Utf8View => string_view_to_decimal_cast::<T>(
            from.as_any().downcast_ref::<StringViewArray>().unwrap(),
            precision,
            scale,
            cast_options,
        )?,
        DataType::Utf8 | DataType::LargeUtf8 => string_to_decimal_cast::<T, Offset>(
            from.as_any()
                .downcast_ref::<GenericStringArray<Offset>>()
                .unwrap(),
            precision,
            scale,
            cast_options,
        )?,
        other => {
            return Err(ArrowError::ComputeError(format!(
                "Cannot cast {other:?} to decimal",
            )));
        }
    };

    Ok(Arc::new(result))
}

pub(crate) fn cast_floating_point_to_decimal<T: ArrowPrimitiveType, D>(
    array: &PrimitiveArray<T>,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: DecimalCast,
{
    let mul = 10_f64.powi(scale as i32);

    if cast_options.safe {
        array
            .unary_opt::<_, D>(|v| {
                single_float_to_decimal::<D>(v.as_(), mul)
                    .filter(|v| D::is_valid_decimal_precision(*v, precision))
            })
            .with_precision_and_scale(precision, scale)
            .map(|a| Arc::new(a) as ArrayRef)
    } else {
        array
            .try_unary::<_, D, _>(|v| {
                let v = single_float_to_decimal::<D>(v.as_(), mul).ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "Cannot cast to {}({}, {}). Overflowing on {:?}",
                        D::PREFIX,
                        precision,
                        scale,
                        v
                    ))
                })?;
                D::validate_decimal_precision(v, precision, scale).map(|()| v)
            })?
            .with_precision_and_scale(precision, scale)
            .map(|a| Arc::new(a) as ArrayRef)
    }
}

/// Cast a single floating point value to a decimal native with the given multiple.
/// Returns `None` if the value cannot be represented with the requested precision.
#[inline(always)]
pub fn single_float_to_decimal<D>(input: f64, mul: f64) -> Option<D::Native>
where
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: DecimalCast,
{
    D::Native::from_f64((mul * input).round())
}

pub(crate) fn cast_decimal_to_integer<D, T>(
    array: &dyn Array,
    base: D::Native,
    scale: i8,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError>
where
    T: ArrowPrimitiveType,
    <T as ArrowPrimitiveType>::Native: NumCast,
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: ToPrimitive,
{
    let array = array.as_primitive::<D>();
    if let Some(conversion) = &cast_options.conversion {
        return conversion_decimal_to_integer::<D, T>(array, base, scale, conversion);
    }

    let div: D::Native = base.pow_checked(scale.unsigned_abs() as u32).map_err(|_| {
        ArrowError::CastError(format!(
            "Cannot cast to {:?}. The scale {} causes overflow.",
            D::PREFIX,
            scale,
        ))
    })?;

    let mut value_builder = PrimitiveBuilder::<T>::with_capacity(array.len());

    if scale < 0 {
        match cast_options.safe {
            true => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array
                            .value(i)
                            .mul_checked(div)
                            .ok()
                            .and_then(<T::Native as NumCast>::from::<D::Native>);
                        value_builder.append_option(v);
                    }
                }
            }
            false => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array.value(i).mul_checked(div)?;

                        let value =
                            <T::Native as NumCast>::from::<D::Native>(v).ok_or_else(|| {
                                ArrowError::CastError(format!(
                                    "value of {:?} is out of range {}",
                                    v,
                                    T::DATA_TYPE
                                ))
                            })?;

                        value_builder.append_value(value);
                    }
                }
            }
        }
    } else {
        match cast_options.safe {
            true => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array
                            .value(i)
                            .div_checked(div)
                            .ok()
                            .and_then(<T::Native as NumCast>::from::<D::Native>);
                        value_builder.append_option(v);
                    }
                }
            }
            false => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array.value(i).div_checked(div)?;

                        let value =
                            <T::Native as NumCast>::from::<D::Native>(v).ok_or_else(|| {
                                ArrowError::CastError(format!(
                                    "value of {:?} is out of range {}",
                                    v,
                                    T::DATA_TYPE
                                ))
                            })?;

                        value_builder.append_value(value);
                    }
                }
            }
        }
    }
    Ok(Arc::new(value_builder.finish()))
}

/// A decimal native type whose wrapping arithmetic Arrow C++ uses as the
/// working width of a decimal to decimal conversion.
pub(crate) trait WrappingDecimal: ArrowNativeTypeOp {
    /// The value, sign extended.
    fn widen(self) -> i256;
    /// The low bits of `value`.
    fn wrap(value: i256) -> Self;
}

impl WrappingDecimal for i32 {
    fn widen(self) -> i256 {
        i256::from_i128(self as i128)
    }
    fn wrap(value: i256) -> Self {
        value.as_i128() as i32
    }
}

impl WrappingDecimal for i64 {
    fn widen(self) -> i256 {
        i256::from_i128(self as i128)
    }
    fn wrap(value: i256) -> Self {
        value.as_i128() as i64
    }
}

impl WrappingDecimal for i128 {
    fn widen(self) -> i256 {
        i256::from_i128(self)
    }
    fn wrap(value: i256) -> Self {
        value.as_i128()
    }
}

impl WrappingDecimal for i256 {
    fn widen(self) -> i256 {
        self
    }
    fn wrap(value: i256) -> Self {
        value
    }
}

/// Converts decimals to the decimal type `to_type` as Arrow C++ does.
///
/// Arrow C++ computes in a working width chosen by the pair of types: the
/// target width for Decimal32 and Decimal256 targets, the wider of the two
/// otherwise. An input wider than the working width keeps its low bits, and
/// the result keeps the low bits of the working value. Without
/// `allow_decimal_truncate` a value that loses digits when rescaled or does
/// not fit the target precision is refused; with it the value is rescaled
/// with wrapping multiplication or truncating division.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first value refused.
pub(crate) fn conversion_decimal_to_decimal(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    use DataType::*;
    let conversion = cast_options
        .conversion
        .as_ref()
        .expect("the dispatcher routes only conversion casts here");
    match (array.data_type(), to_type) {
        (_, Decimal256(p, s)) => rescale_from::<Decimal256Type, i256>(array, *p, *s, conversion),
        (_, Decimal32(p, s)) => rescale_from::<Decimal32Type, i32>(array, *p, *s, conversion),
        (Decimal32(..) | Decimal64(..), Decimal64(p, s)) => {
            rescale_from::<Decimal64Type, i64>(array, *p, *s, conversion)
        }
        (Decimal128(..), Decimal64(p, s)) => {
            rescale_from::<Decimal64Type, i128>(array, *p, *s, conversion)
        }
        (Decimal256(..), Decimal64(p, s)) => {
            rescale_from::<Decimal64Type, i256>(array, *p, *s, conversion)
        }
        (Decimal256(..), Decimal128(p, s)) => {
            rescale_from::<Decimal128Type, i256>(array, *p, *s, conversion)
        }
        (_, Decimal128(p, s)) => rescale_from::<Decimal128Type, i128>(array, *p, *s, conversion),
        _ => unreachable!("the dispatcher routes only decimal to decimal casts here"),
    }
}

fn rescale_from<O, W>(
    array: &dyn Array,
    precision: u8,
    scale: i8,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError>
where
    O: DecimalType,
    O::Native: WrappingDecimal,
    W: WrappingDecimal,
{
    match array.data_type() {
        DataType::Decimal32(_, s) => {
            rescale::<Decimal32Type, O, W>(array.as_primitive(), *s, precision, scale, conversion)
        }
        DataType::Decimal64(_, s) => {
            rescale::<Decimal64Type, O, W>(array.as_primitive(), *s, precision, scale, conversion)
        }
        DataType::Decimal128(_, s) => {
            rescale::<Decimal128Type, O, W>(array.as_primitive(), *s, precision, scale, conversion)
        }
        DataType::Decimal256(_, s) => {
            rescale::<Decimal256Type, O, W>(array.as_primitive(), *s, precision, scale, conversion)
        }
        other => unreachable!("{other} is not a decimal type"),
    }
}

fn rescale<I, O, W>(
    array: &PrimitiveArray<I>,
    in_scale: i8,
    precision: u8,
    scale: i8,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError>
where
    I: DecimalType,
    I::Native: WrappingDecimal,
    O: DecimalType,
    O::Native: WrappingDecimal,
    W: WrappingDecimal,
{
    let ten = W::wrap(i256::from_i128(10));
    let delta = scale as i32 - in_scale as i32;
    let multiplier = ten.pow_wrapping(delta.unsigned_abs());
    let bound = ten.pow_wrapping(precision as u32);
    let result = array.try_unary::<_, O, _>(|v| {
        let v = W::wrap(v.widen());
        let v = if conversion.allow_decimal_truncate {
            if delta > 0 {
                v.mul_wrapping(multiplier)
            } else {
                v.div_wrapping(multiplier)
            }
        } else {
            let rescaled = if delta < 0 {
                (v.mod_wrapping(multiplier) == W::ZERO).then(|| v.div_wrapping(multiplier))
            } else {
                let product = v.mul_wrapping(multiplier);
                let overflowed = if v.is_lt(W::ZERO) {
                    product.is_gt(v)
                } else {
                    product.is_lt(v)
                };
                (!overflowed).then_some(product)
            };
            let rescaled = rescaled.ok_or_else(|| {
                ArrowError::CastError(format!(
                    "Rescaling decimal value {v:?} from scale {in_scale} to scale {scale} would cause data loss"
                ))
            })?;
            let magnitude = if rescaled.is_lt(W::ZERO) {
                rescaled.neg_wrapping()
            } else {
                rescaled
            };
            if !magnitude.is_lt(bound) {
                return Err(ArrowError::CastError(format!(
                    "Decimal value {rescaled:?} does not fit in precision {precision}"
                )));
            }
            rescaled
        };
        Ok(O::Native::wrap(v.widen()))
    })?;
    Ok(Arc::new(result.with_precision_and_scale(precision, scale)?))
}

/// Converts decimals to integers as Arrow C++ does.
///
/// Without `allow_decimal_truncate` a value is rescaled to scale zero and
/// refused when that loses digits or overflows; with it a positive scale
/// divides the fraction away and a negative scale multiplies with wrapping.
/// The integer is then refused outside the target range unless
/// `allow_int_overflow`, which keeps its low bits instead.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first value that loses digits or
/// does not fit the target.
fn conversion_decimal_to_integer<D, T>(
    array: &PrimitiveArray<D>,
    base: D::Native,
    scale: i8,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NumCast,
    D: DecimalType + ArrowPrimitiveType,
    D::Native: ToPrimitive,
{
    let multiplier = base.pow_wrapping(scale.unsigned_abs() as u32);
    let rescale = |v: D::Native| -> Result<D::Native, ArrowError> {
        let lost = || {
            ArrowError::CastError(format!(
                "Rescaling decimal value {v:?} with scale {scale} would cause data loss"
            ))
        };
        if scale > 0 {
            let whole = v.div_wrapping(multiplier);
            if !conversion.allow_decimal_truncate && v.mod_wrapping(multiplier) != D::Native::ZERO {
                return Err(lost());
            }
            Ok(whole)
        } else {
            let whole = v.mul_wrapping(multiplier);
            let overflowed = if v.is_lt(D::Native::ZERO) {
                whole.is_gt(v)
            } else {
                whole.is_lt(v)
            };
            if !conversion.allow_decimal_truncate && overflowed {
                return Err(lost());
            }
            Ok(whole)
        }
    };
    let result = array.try_unary::<_, T, _>(|v| {
        let whole = rescale(v)?;
        match <T::Native as NumCast>::from::<D::Native>(whole) {
            Some(value) => Ok(value),
            None if conversion.allow_int_overflow => Ok(T::Native::usize_as(whole.as_usize())),
            None => Err(ArrowError::CastError(format!(
                "Integer value {whole:?} not in range of {}",
                T::DATA_TYPE
            ))),
        }
    })?;
    Ok(Arc::new(result))
}

/// Cast a decimal array to a floating point array.
///
/// Conversion is lossy and follows standard floating point semantics. Values
/// that exceed the representable range become `INFINITY` or `-INFINITY` without
/// returning an error.
pub(crate) fn cast_decimal_to_float<D: DecimalType, T: ArrowPrimitiveType, F>(
    array: &dyn Array,
    op: F,
) -> Result<ArrayRef, ArrowError>
where
    F: Fn(D::Native) -> T::Native,
{
    let array = array.as_primitive::<D>();
    let array = array.unary::<_, T>(op);
    Ok(Arc::new(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(deprecated)]
    fn test_parse_string_to_decimal_native() {
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>("123.456", 2).unwrap(),
            12346
        );
        // The value is checked against the maximum precision of the type, not
        // against any target precision
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>(&"9".repeat(38), 0).unwrap(),
            10_i128.pow(38) - 1
        );
        assert!(
            parse_string_to_decimal_native::<Decimal128Type>(&i128::MAX.to_string(), 0).is_err()
        );
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>("abc", 2)
                .unwrap_err()
                .to_string(),
            "Invalid argument error: Invalid decimal format: \"abc\""
        );
        assert_eq!(
            parse_string_to_decimal_native::<Decimal32Type>("1", 10)
                .unwrap_err()
                .to_string(),
            "Invalid argument error: Cannot convert 1 to Decimal32: Overflow"
        );
    }

    #[test]
    fn test_rescale_decimal_upscale_within_precision() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(
            12_345_i128, // 123.45 with scale 2
            5,
            2,
            8,
            5,
        );
        assert_eq!(result, Some(12_345_000_i128));
    }

    #[test]
    fn test_rescale_decimal_downscale_rounds_half_away_from_zero() {
        let positive = rescale_decimal::<Decimal128Type, Decimal128Type>(
            1_050_i128, // 1.050 with scale 3
            5, 3, 5, 1,
        );
        assert_eq!(positive, Some(11_i128)); // 1.1 with scale 1

        let negative = rescale_decimal::<Decimal128Type, Decimal128Type>(
            -1_050_i128, // -1.050 with scale 3
            5,
            3,
            5,
            1,
        );
        assert_eq!(negative, Some(-11_i128)); // -1.1 with scale 1
    }

    #[test]
    fn test_rescale_decimal_downscale_large_delta_returns_zero() {
        let result = rescale_decimal::<Decimal32Type, Decimal32Type>(12_345_i32, 9, 9, 9, 4);
        assert_eq!(result, Some(0_i32));
    }

    #[test]
    fn test_rescale_decimal_upscale_overflow_returns_none() {
        let result = rescale_decimal::<Decimal32Type, Decimal32Type>(9_999_i32, 4, 0, 5, 2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_rescale_decimal_invalid_input_precision_scale_returns_none() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(123_i128, 39, 39, 38, 38);
        assert_eq!(result, None);
    }

    #[test]
    fn test_rescale_decimal_invalid_output_precision_scale_returns_none() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(123_i128, 38, 38, 39, 39);
        assert_eq!(result, None);
    }
}

/// A floating point type Arrow C++ converts to decimals with `FromReal`.
pub(crate) trait CppReal:
    ArrowNativeType + PartialOrd + std::ops::Mul<Output = Self> + std::ops::Neg<Output = Self>
{
    /// Bits of the significand, including the implicit bit.
    const MANTISSA_BITS: i32;
    /// `ceil(log10(2 ^ MANTISSA_BITS))`.
    const MANTISSA_DIGITS: i32;
    /// `10 ^ exp`, correctly rounded, as Arrow C++ tabulates it.
    fn power_of_ten(exp: i32) -> Self;
    /// The value rounded to an integer, half to even.
    fn nearbyint(self) -> Self;
    /// The value, exactly.
    fn widen(self) -> f64;
}

impl CppReal for f32 {
    const MANTISSA_BITS: i32 = 24;
    const MANTISSA_DIGITS: i32 = 8;
    fn power_of_ten(exp: i32) -> Self {
        format!("1e{exp}")
            .parse()
            .expect("a power of ten is a valid float literal")
    }
    fn nearbyint(self) -> Self {
        self.round_ties_even()
    }
    fn widen(self) -> f64 {
        self as f64
    }
}

impl CppReal for f64 {
    const MANTISSA_BITS: i32 = 53;
    const MANTISSA_DIGITS: i32 = 16;
    fn power_of_ten(exp: i32) -> Self {
        format!("1e{exp}")
            .parse()
            .expect("a power of ten is a valid float literal")
    }
    fn nearbyint(self) -> Self {
        self.round_ties_even()
    }
    fn widen(self) -> f64 {
        self
    }
}

/// `ceil(log2(10 ^ k))` for `k` in `0..=76`, as Arrow C++ tabulates it.
const CEIL_LOG2_POWERS_OF_TEN: [i32; 77] = [
    0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67, 70, 74, 77,
    80, 84, 87, 90, 94, 97, 100, 103, 107, 110, 113, 117, 120, 123, 127, 130, 133, 137, 140, 143,
    147, 150, 153, 157, 160, 163, 167, 170, 173, 177, 180, 183, 187, 190, 193, 196, 200, 203, 206,
    210, 213, 216, 220, 223, 226, 230, 233, 236, 240, 243, 246, 250, 253,
];

/// The normalized fraction in `[0.5, 1)` and the exponent of positive finite `value`.
fn frexp(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (fraction, exponent) = frexp(value * 2f64.powi(54));
        return (fraction, exponent - 54);
    }
    let fraction = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (fraction, exponent - 1022)
}

/// Non-negative `x` shifted right by `bits`, rounded half to even.
fn rounded_right_shift(x: i256, bits: i32) -> i256 {
    use num_traits::{WrappingShl, WrappingShr};
    if bits == 0 {
        return x;
    }
    if bits >= 256 {
        return i256::ZERO;
    }
    let quotient = x.wrapping_shr(bits as u32);
    let remainder = x.wrapping_sub(quotient.wrapping_shl(bits as u32));
    let half = i256::ONE.wrapping_shl(bits as u32 - 1);
    if remainder > half || (remainder == half && quotient.as_i128() & 1 == 1) {
        quotient.wrapping_add(i256::ONE)
    } else {
        quotient
    }
}

/// Converts floating point values to the decimal type `D` as Arrow C++
/// `Decimal*::FromReal` does: exactly, rounding half to even, through the
/// binary significand for Decimal64, Decimal128 and Decimal256 with a
/// non-negative scale, and approximately in floating point otherwise. With
/// `allow_decimal_truncate` a value that cannot be converted becomes zero.
///
/// # Errors
///
/// Returns [`ArrowError::CastError`] for the first non-finite value or value
/// that does not fit the target precision, without `allow_decimal_truncate`.
pub(crate) fn conversion_real_to_decimal<R, D>(
    array: &PrimitiveArray<R>,
    precision: u8,
    scale: i8,
    conversion: &ConversionOptions,
) -> Result<ArrayRef, ArrowError>
where
    R: ArrowPrimitiveType,
    R::Native: CppReal + std::fmt::Display,
    D: DecimalType,
    D::Native: WrappingDecimal,
{
    use num_traits::WrappingShl;
    let max_precision = D::MAX_PRECISION as i32;
    let (p, s) = (precision as i32, scale as i32);
    let narrow_approx = max_precision <= Decimal64Type::MAX_PRECISION as i32;
    let real_scale = R::Native::power_of_ten(s);
    let real_max = R::Native::power_of_ten(p);
    let limit = R::Native::power_of_ten(p - s.max(0));
    let ten = |exp: i32| i256::from_i128(10).wrapping_pow(exp as u32);
    let width = |x: i256| D::Native::wrap(x).widen();
    let max_abs = ten(p);
    let overflow = |real: R::Native| {
        ArrowError::CastError(format!(
            "Cannot convert {real} to {}({precision}, {scale}): overflow",
            D::PREFIX
        ))
    };
    let approx = |real: R::Native| -> Result<i256, ArrowError> {
        let x = if narrow_approx {
            let x = (real * real_scale).nearbyint();
            if x >= real_max {
                return Err(overflow(real));
            }
            x.widen()
        } else {
            let x = (real.widen() * f64::power_of_ten(s)).nearbyint();
            if x >= f64::power_of_ten(p) {
                return Err(overflow(real));
            }
            x
        };
        i256::from_f64(x).ok_or_else(|| overflow(real))
    };
    let from_positive = |real: R::Native| -> Result<i256, ArrowError> {
        if max_precision == Decimal32Type::MAX_PRECISION as i32 || s < 0 {
            return approx(real);
        }
        if real > limit {
            return Err(overflow(real));
        }
        let (fraction, exponent) = frexp(real.widen());
        let mantissa = (fraction * 2f64.powi(R::Native::MANTISSA_BITS)) as u64;
        let k = exponent - R::Native::MANTISSA_BITS;
        let mut x = i256::from_i128(mantissa as i128);
        if k < 0 {
            let mut right_shift_by = -k;
            let mut mul_by_ten_to = s;
            let safe = max_precision - R::Native::MANTISSA_DIGITS;
            if mul_by_ten_to <= safe {
                x = width(x.wrapping_mul(ten(mul_by_ten_to)));
                x = width(rounded_right_shift(x, right_shift_by));
            } else {
                x = width(x.wrapping_mul(ten(safe)));
                mul_by_ten_to -= safe;
                let step = (max_precision - p).max(1);
                let (mut total_exp, mut total_shift) = (0, 0);
                while mul_by_ten_to > 0 && right_shift_by > 0 {
                    let exp = mul_by_ten_to.min(step);
                    total_exp += exp;
                    let bits = right_shift_by
                        .min(CEIL_LOG2_POWERS_OF_TEN[total_exp as usize] - total_shift);
                    total_shift += bits;
                    x = width(rounded_right_shift(x, bits));
                    right_shift_by -= bits;
                    x = width(x.wrapping_mul(ten(exp)));
                    mul_by_ten_to -= exp;
                }
                if mul_by_ten_to > 0 {
                    x = width(x.wrapping_mul(ten(mul_by_ten_to)));
                }
                if right_shift_by > 0 {
                    x = width(rounded_right_shift(x, right_shift_by));
                }
            }
        } else {
            x = width(x.wrapping_mul(ten(s)));
            x = width(x.wrapping_shl(k as u32));
        }
        if x.wrapping_abs() >= max_abs {
            return Err(overflow(real));
        }
        Ok(x)
    };
    let zero = R::Native::usize_as(0);
    let from_real = |real: R::Native| -> Result<i256, ArrowError> {
        if !real.widen().is_finite() {
            Err(ArrowError::CastError(format!(
                "Cannot convert {real} to {}",
                D::PREFIX
            )))
        } else if real == zero {
            Ok(i256::ZERO)
        } else if real < zero {
            from_positive(-real).map(|x| x.wrapping_neg())
        } else {
            from_positive(real)
        }
    };
    array
        .try_unary::<_, D, _>(|real| match from_real(real) {
            Ok(x) => Ok(D::Native::wrap(x)),
            Err(_) if conversion.allow_decimal_truncate => Ok(D::Native::usize_as(0)),
            Err(error) => Err(error),
        })?
        .with_precision_and_scale(precision, scale)
        .map(|array| Arc::new(array) as ArrayRef)
}
