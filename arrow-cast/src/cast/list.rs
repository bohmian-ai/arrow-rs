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

/// Converts a non-list array to a list array where every element is a single element
/// list. `NULL`s in the original array become `[NULL]` (i.e. output list array
/// contains no nulls since it wraps all input nulls in a single element list).
///
/// For example: `Int32([1, NULL, 2]) -> List<Int32>([[1], [NULL], [2]])`
pub(crate) fn cast_values_to_list<O: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    if array.len() > O::MAX_OFFSET {
        return Err(ArrowError::ComputeError(format!(
            "Offset overflow when casting from {} to {}",
            array.data_type(),
            to.data_type()
        )));
    }
    let values = cast_with_context(array, to.data_type(), cast_options)?;
    let offsets = OffsetBuffer::from_repeated_length(1, values.len());
    let list = GenericListArray::<O>::try_new(to.clone(), offsets, values, None)?;
    Ok(Arc::new(list))
}

/// Same as [`cast_values_to_list`] but output list view array.
pub(crate) fn cast_values_to_list_view<O: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    if array.len() > O::MAX_OFFSET {
        return Err(ArrowError::ComputeError(format!(
            "Offset overflow when casting from {} to {}",
            array.data_type(),
            to.data_type()
        )));
    }
    let values = cast_with_context(array, to.data_type(), cast_options)?;
    let offsets = (0..values.len())
        .map(|index| O::usize_as(index))
        .collect::<Vec<O>>();
    let list = GenericListViewArray::<O>::try_new(
        to.clone(),
        offsets.into(),
        vec![O::one(); values.len()].into(),
        values,
        None,
    )?;
    Ok(Arc::new(list))
}

/// Cast fixed size list array to inner values type, essentially flattening the
/// lists.
///
/// For example: `FixedSizeList<Int32, 2>([[1, 2], [3, 4]]) -> Int32([1, 2, 3, 4])`
pub(crate) fn cast_single_element_fixed_size_list_to_values(
    array: &dyn Array,
    to: &DataType,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let values = array.as_fixed_size_list().values();
    cast_with_context(values, to, cast_options)
}

fn cast_fixed_size_list_to_list_inner<OffsetSize: OffsetSizeTrait, const IS_LIST_VIEW: bool>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_fixed_size_list();
    let DataType::FixedSizeList(inner_field, size) = array.data_type() else {
        unreachable!()
    };
    let array = if to != inner_field {
        // To transform inner type, can first cast to FSL with new inner type.
        let fsl_to = DataType::FixedSizeList(to.clone(), *size);
        let array = cast_with_context(array, &fsl_to, cast_options)?;
        array.as_fixed_size_list().clone()
    } else {
        array.clone()
    };
    if IS_LIST_VIEW {
        let list: GenericListViewArray<OffsetSize> = array.into();
        Ok(Arc::new(list))
    } else {
        let list: GenericListArray<OffsetSize> = array.into();
        Ok(Arc::new(list))
    }
}

/// Cast fixed size list arrays to list arrays, maintaining the lengths of the inner
/// lists.
///
/// For example: `FixedSizeList<Int32, 2>([[1, 2], [3, 4]]) -> List<Int32>([[1, 2], [3, 4]])`
pub(crate) fn cast_fixed_size_list_to_list<OffsetSize: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    cast_fixed_size_list_to_list_inner::<OffsetSize, false>(array, to, cast_options)
}

/// Same as [`cast_fixed_size_list_to_list`] but output list view array.
pub(crate) fn cast_fixed_size_list_to_list_view<OffsetSize: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    cast_fixed_size_list_to_list_inner::<OffsetSize, true>(array, to, cast_options)
}

/// Cast list to fixed size list array. If any inner list size does not match the
/// size of the output fixed size list array, depending on `cast_options` we either
/// output `NULL` for that element (safe) or raise an error.
pub(crate) fn cast_list_to_fixed_size_list<OffsetSize>(
    array: &dyn Array,
    field: &FieldRef,
    size: i32,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError>
where
    OffsetSize: OffsetSizeTrait,
{
    let array = array.as_list::<OffsetSize>();

    let cap = array.len() * size as usize;

    let mut null_builder = NullBufferBuilder::new(array.len());
    if let Some(nulls) = array.nulls().filter(|b| b.null_count() > 0) {
        null_builder.append_buffer(nulls);
    } else {
        null_builder.append_n_non_nulls(array.len());
    }

    // Whether the resulting array may contain null lists
    let nullable = cast_options.safe || array.null_count() != 0;
    // Nulls in FixedSizeListArray take up space and so we must pad the values
    let values = array.values().to_data();
    let mut mutable = MutableArrayData::new(vec![&values], nullable, cap);
    // The values of a sliced list start at its first offset, not at zero.
    let first_pos = array.offsets()[0].as_usize();
    // The end position in values of the last incorrectly-sized list slice
    let mut last_pos = first_pos;
    // Whether any slice was replaced by nulls, so the values must be copied
    let mut padded = false;

    for (idx, w) in array.offsets().windows(2).enumerate() {
        let start_pos = w[0].as_usize();
        let end_pos = w[1].as_usize();
        let len = end_pos - start_pos;

        if len != size as usize {
            if cast_options.safe || array.is_null(idx) {
                if last_pos != start_pos {
                    // Extend with valid slices
                    mutable
                        .try_extend(0, last_pos, start_pos)
                        .map_err(|e| ArrowError::CastError(e.to_string()))?;
                }
                // Pad this slice with nulls
                mutable
                    .try_extend_nulls(size as _)
                    .map_err(|e| ArrowError::CastError(e.to_string()))?;
                null_builder.set_bit(idx, false);
                // Set last_pos to the end of this slice's values
                last_pos = end_pos;
                padded = true;
            } else {
                return Err(ArrowError::CastError(format!(
                    "Cannot cast to FixedSizeList({size}): value at index {idx} has length {len}",
                )));
            }
        }
    }

    let values = if padded {
        // Remaining slices were all correct length
        let end_pos = array.offsets()[array.len()].as_usize();
        mutable
            .try_extend(0, last_pos, end_pos)
            .map_err(|e| ArrowError::CastError(e.to_string()))?;
        make_array(mutable.freeze())
    } else {
        // All slices were the correct length
        array.values().slice(first_pos, cap)
    };

    // Cast the inner values if necessary
    let values = cast_field_with_context(
        values.as_ref(),
        item(array.data_type()),
        field,
        cast_options,
    )?;

    let nulls = null_builder.build();
    // Degenerate case where we may lose length information if there isn't a null
    // buffer to infer length from
    let array = if size == 0 && nulls.is_none() {
        FixedSizeListArray::try_new_with_length(field.clone(), size, values, nulls, array.len())?
    } else {
        FixedSizeListArray::try_new(field.clone(), size, values, nulls)?
    };
    Ok(Arc::new(array))
}

/// Same as [`cast_list_to_fixed_size_list`] but for list view arrays.
pub(crate) fn cast_list_view_to_fixed_size_list<O: OffsetSizeTrait>(
    array: &dyn Array,
    field: &FieldRef,
    size: i32,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let array = array.as_list_view::<O>();

    let mut null_builder = NullBufferBuilder::new(array.len());
    if let Some(nulls) = array.nulls().filter(|b| b.null_count() > 0) {
        null_builder.append_buffer(nulls);
    } else {
        null_builder.append_n_non_nulls(array.len());
    }

    let nullable = cast_options.safe || array.null_count() != 0;
    let values = array.values().to_data();
    let cap = array.len() * size as usize;
    let mut mutable = MutableArrayData::new(vec![&values], nullable, cap);

    for idx in 0..array.len() {
        let offset = array.value_offset(idx).as_usize();
        let len = array.value_size(idx).as_usize();

        if len != size as usize {
            // Nulls in FixedSizeListArray take up space and so we must pad the values
            if cast_options.safe || array.is_null(idx) {
                mutable
                    .try_extend_nulls(size as _)
                    .map_err(|e| ArrowError::CastError(e.to_string()))?;
                null_builder.set_bit(idx, false);
            } else {
                return Err(ArrowError::CastError(format!(
                    "Cannot cast to FixedSizeList({size}): value at index {idx} has length {len}",
                )));
            }
        } else {
            mutable
                .try_extend(0, offset, offset + len)
                .map_err(|e| ArrowError::CastError(e.to_string()))?;
        }
    }

    let values = make_array(mutable.freeze());
    let values = cast_field_with_context(
        values.as_ref(),
        item(array.data_type()),
        field,
        cast_options,
    )?;

    let nulls = null_builder.build();
    // Degenerate case where we may lose length information if there isn't a null
    // buffer to infer length from
    let array = if size == 0 && nulls.is_none() {
        FixedSizeListArray::try_new_with_length(field.clone(), size, values, nulls, array.len())?
    } else {
        FixedSizeListArray::try_new(field.clone(), size, values, nulls)?
    };
    Ok(Arc::new(array))
}

/// The item field of the list type `data_type`, which carries the item's
/// extension type.
fn item(data_type: &DataType) -> &Field {
    match data_type {
        DataType::List(item)
        | DataType::LargeList(item)
        | DataType::ListView(item)
        | DataType::LargeListView(item)
        | DataType::FixedSizeList(item, _) => item,
        _ => unreachable!("a list array has a list type"),
    }
}

/// Casting between list arrays of same offset size; we cast only the inner type.
pub(crate) fn cast_list_values<O: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let list = array.as_list::<O>();
    let values = cast_field_with_context(list.values(), item(list.data_type()), to, cast_options)?;
    Ok(Arc::new(GenericListArray::<O>::try_new(
        to.clone(),
        list.offsets().clone(),
        values,
        list.nulls().cloned(),
    )?))
}

/// Casting between list view arrays of same offset size; we cast only the inner type.
pub(crate) fn cast_list_view_values<O: OffsetSizeTrait>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let list = array.as_list_view::<O>();
    let values = cast_with_context(list.values(), to.data_type(), cast_options)?;
    Ok(Arc::new(GenericListViewArray::<O>::try_new(
        to.clone(),
        list.offsets().clone(),
        list.sizes().clone(),
        values,
        list.nulls().cloned(),
    )?))
}

/// Casting between list arrays of different offset size (e.g. List -> LargeList)
pub(crate) fn cast_list<I: OffsetSizeTrait, O: OffsetSizeTrait>(
    array: &dyn Array,
    field: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let list = array.as_list::<I>();
    let values = list.values();
    let offsets = list.offsets();
    let nulls = list.nulls().cloned();

    if offsets.last().as_usize() > O::MAX_OFFSET {
        return Err(ArrowError::ComputeError(format!(
            "Offset overflow when casting from {} to {}",
            array.data_type(),
            field.data_type()
        )));
    }

    // Recursively cast values
    let values = cast_field_with_context(values, item(list.data_type()), field, cast_options)?;
    let offsets: Vec<_> = offsets.iter().map(|x| O::usize_as(x.as_usize())).collect();

    // Safety: valid offsets and checked for overflow
    let offsets = unsafe { OffsetBuffer::new_unchecked(offsets.into()) };

    Ok(Arc::new(GenericListArray::<O>::try_new(
        field.clone(),
        offsets,
        values,
        nulls,
    )?))
}

/// Casting list view arrays to list.
pub(crate) fn cast_list_view_to_list<I, O>(
    array: &dyn Array,
    to: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError>
where
    I: OffsetSizeTrait,
    // We need ArrowPrimitiveType here to be able to create indices array for the
    // take kernel.
    O: ArrowPrimitiveType,
    O::Native: OffsetSizeTrait,
{
    let list_view = array.as_list_view::<I>();
    let list_view_offsets = list_view.offsets();
    let sizes = list_view.sizes();

    let mut take_indices: Vec<O::Native> = Vec::with_capacity(list_view.values().len());
    let mut offsets: Vec<O::Native> = Vec::with_capacity(list_view.len() + 1);
    use num_traits::Zero;
    offsets.push(O::Native::zero());

    for i in 0..list_view.len() {
        if list_view.is_null(i) {
            offsets.push(O::Native::usize_as(take_indices.len()));
            continue;
        }

        let offset = list_view_offsets[i].as_usize();
        let size = sizes[i].as_usize();

        for value_index in offset..offset + size {
            take_indices.push(O::Native::usize_as(value_index));
        }

        // Must guard all cases since ListView<i32> can overflow List<i32>
        // e.g. if offsets of [0, 0, 0] and sizes [i32::MAX, i32::MAX, i32::MAX]
        if take_indices.len() > O::Native::MAX_OFFSET {
            return Err(ArrowError::ComputeError(format!(
                "Offset overflow when casting from {} to {}",
                array.data_type(),
                to.data_type()
            )));
        }
        offsets.push(O::Native::usize_as(take_indices.len()));
    }

    // Form a contiguous values array
    let take_indices = PrimitiveArray::<O>::from_iter_values(take_indices);
    let values = arrow_select::take::take(list_view.values(), &take_indices, None)?;
    let values = cast_field_with_context(&values, item(list_view.data_type()), to, cast_options)?;

    Ok(Arc::new(GenericListArray::<O::Native>::try_new(
        to.clone(),
        OffsetBuffer::new(offsets.into()),
        values,
        list_view.nulls().cloned(),
    )?))
}

/// Casting between list view arrays of different offset size (e.g. ListView -> LargeListView)
pub(crate) fn cast_list_view<I: OffsetSizeTrait, O: OffsetSizeTrait>(
    array: &dyn Array,
    to_field: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let list_view = array.as_list_view::<I>();

    // Recursively cast values
    let values = cast_with_context(list_view.values(), to_field.data_type(), cast_options)?;

    let offsets = list_view
        .offsets()
        .iter()
        .map(|offset| {
            let offset = offset.as_usize();
            if offset > O::MAX_OFFSET {
                return Err(ArrowError::ComputeError(format!(
                    "Offset overflow when casting from {} to {}",
                    array.data_type(),
                    to_field.data_type()
                )));
            }
            Ok(O::usize_as(offset))
        })
        .collect::<Result<Vec<O>, _>>()?;
    let sizes = list_view
        .sizes()
        .iter()
        .map(|size| {
            let size = size.as_usize();
            if size > O::MAX_OFFSET {
                return Err(ArrowError::ComputeError(format!(
                    "Offset overflow when casting from {} to {}",
                    array.data_type(),
                    to_field.data_type()
                )));
            }
            Ok(O::usize_as(size))
        })
        .collect::<Result<Vec<O>, _>>()?;
    Ok(Arc::new(GenericListViewArray::<O>::try_new(
        to_field.clone(),
        offsets.into(),
        sizes.into(),
        values,
        list_view.nulls().cloned(),
    )?))
}

/// Casting list arrays to list view.
pub(crate) fn cast_list_to_list_view<I: OffsetSizeTrait, O: OffsetSizeTrait>(
    array: &dyn Array,
    to_field: &FieldRef,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let list = array.as_list::<I>();
    let (_field, offsets, values, nulls) = list.clone().into_parts();

    let len = offsets.len() - 1;
    let mut sizes = Vec::with_capacity(len);
    let mut view_offsets = Vec::with_capacity(len);
    for (i, offset) in offsets.iter().enumerate().take(len) {
        let offset = offset.as_usize();
        let size = offsets[i + 1].as_usize() - offset;

        if offset > O::MAX_OFFSET || size > O::MAX_OFFSET {
            return Err(ArrowError::ComputeError(format!(
                "Offset overflow when casting from {} to {}",
                array.data_type(),
                to_field.data_type()
            )));
        }

        view_offsets.push(O::usize_as(offset));
        sizes.push(O::usize_as(size));
    }
    let values = cast_with_context(&values, to_field.data_type(), cast_options)?;
    let array = GenericListViewArray::<O>::new(
        to_field.clone(),
        view_offsets.into(),
        sizes.into(),
        values,
        nulls,
    );
    Ok(Arc::new(array))
}

/// A list, large list, list view, large list view or map array as Arrow C++
/// reads it when it casts one as a list.
///
/// C++ reads a list view's offsets as list offsets and ignores its sizes,
/// reading one offset past its length. The Rust array holds no such offset,
/// so the end of the last view stands for it.
struct CppList {
    /// The `len + 1` offsets C++ reads.
    offsets: Vec<i64>,
    /// Whether the array is a slice of a longer one: C++ selects the values
    /// of a list with a nonzero array offset and converts every value of one
    /// with a zero offset. A Rust array keeps no array offset, so a slice is
    /// recognized by an offsets buffer that starts inside its allocation.
    sliced: bool,
    /// The child array, unsliced.
    values: ArrayRef,
}

impl CppList {
    /// Reads `array`, which is of a list-like type.
    fn new(array: &dyn Array) -> Self {
        fn list<O: OffsetSizeTrait>(array: &GenericListArray<O>) -> CppList {
            CppList {
                offsets: array
                    .offsets()
                    .iter()
                    .map(|o| o.as_usize() as i64)
                    .collect(),
                sliced: array.offsets().inner().inner().ptr_offset() != 0,
                values: array.values().clone(),
            }
        }
        fn view<O: OffsetSizeTrait>(array: &GenericListViewArray<O>) -> CppList {
            let mut offsets: Vec<i64> = array
                .offsets()
                .iter()
                .map(|o| o.as_usize() as i64)
                .collect();
            let end = array.len().checked_sub(1).map_or(0, |last| {
                (array.offsets()[last].as_usize() + array.sizes()[last].as_usize()) as i64
            });
            offsets.push(end);
            CppList {
                offsets,
                sliced: array.offsets().inner().ptr_offset() != 0,
                values: array.values().clone(),
            }
        }
        match array.data_type() {
            DataType::List(_) => list(array.as_list::<i32>()),
            DataType::LargeList(_) => list(array.as_list::<i64>()),
            DataType::ListView(_) => view(array.as_list_view::<i32>()),
            DataType::LargeListView(_) => view(array.as_list_view::<i64>()),
            DataType::Map(..) => {
                let map = array.as_map();
                CppList {
                    offsets: map.offsets().iter().map(|o| *o as i64).collect(),
                    sliced: map.offsets().inner().inner().ptr_offset() != 0,
                    values: Arc::new(map.entries().clone()),
                }
            }
            _ => unreachable!("only list-like arrays are cast as lists"),
        }
    }

    /// The offsets and values of the list C++ `CastList::HandleOffsets`
    /// makes: a slice keeps the values from its first offset, as many as its
    /// last offset counts, and rebases its offsets on the first.
    ///
    /// # Errors
    ///
    /// [`ArrowError::CastError`] (C++ `Invalid`) when the last offset exceeds
    /// `max`, the largest offset of the target, and when the offsets C++
    /// reads do not delimit lists of the values, as the offsets of list views
    /// that are not in list order may not.
    fn select(
        self,
        from: &DataType,
        to: &DataType,
        max: i64,
    ) -> Result<(Vec<i64>, ArrayRef), ArrowError> {
        let Self {
            mut offsets,
            sliced,
            mut values,
        } = self;
        let end = offsets[offsets.len() - 1];
        if end > max {
            return Err(ArrowError::CastError(format!(
                "Array of type {from} too large to convert to {to}"
            )));
        }
        let first = offsets[0];
        let ordered = |offsets: &[i64], len: usize| {
            offsets.windows(2).all(|w| w[0] <= w[1]) && offsets[offsets.len() - 1] as usize <= len
        };
        if sliced && first as usize <= values.len() {
            let first_index = first as usize;
            values = values.slice(first_index, (values.len() - first_index).min(end as usize));
            offsets.iter_mut().for_each(|offset| *offset -= first);
        }
        if !ordered(&offsets, values.len()) {
            return Err(ArrowError::CastError(format!(
                "The offsets Arrow C++ reads from the {from} array do not delimit lists of its values"
            )));
        }
        Ok((offsets, values))
    }
}

/// Casts a list, large list, list view or large list view array to a list or
/// large list, or a map array to a list, large list or map, as Arrow C++
/// `CastList` and `CastMap` do: the offsets select values as
/// [`CppList::select`] describes, then the values convert. A map converts its
/// keys and values by position into the two fields of the target's struct,
/// whose entries have no nulls, and takes the target's sort flag.
///
/// # Errors
///
/// The errors of [`CppList::select`], [`ArrowError::SchemaError`] (C++
/// `TypeError`) for a map cast to a list of other than a struct of two
/// fields, and any error of converting the values.
pub(crate) fn conversion_list(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let from_type = array.data_type();
    let max = match to_type {
        DataType::LargeList(_) => i64::MAX,
        _ => i32::MAX as i64,
    };
    let (offsets, values) = CppList::new(array).select(from_type, to_type, max)?;
    let (DataType::List(target) | DataType::LargeList(target) | DataType::Map(target, _)) = to_type
    else {
        unreachable!("the dispatcher casts lists to lists and maps")
    };
    let values = match from_type {
        DataType::Map(..) => {
            let DataType::Struct(fields) = target.data_type() else {
                return Err(map_entries(to_type));
            };
            let [key, value] = fields.iter().collect::<Vec<_>>()[..] else {
                return Err(map_entries(to_type));
            };
            let entries = values.as_struct();
            let from = entries.fields();
            let keys = cast_field_with_context(entries.column(0), &from[0], key, cast_options)?;
            let values = cast_field_with_context(entries.column(1), &from[1], value, cast_options)?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                vec![keys, values],
                None,
            )?)
        }
        _ => cast_field_with_context(&values, item(from_type), target, cast_options)?,
    };
    let nulls = array.nulls().cloned();
    Ok(match to_type {
        DataType::LargeList(_) => Arc::new(LargeListArray::try_new(
            target.clone(),
            OffsetBuffer::new(offsets.into()),
            values,
            nulls,
        )?),
        DataType::List(_) => Arc::new(ListArray::try_new(
            target.clone(),
            OffsetBuffer::new(offsets.iter().map(|o| *o as i32).collect()),
            values,
            nulls,
        )?),
        DataType::Map(_, sorted) => Arc::new(MapArray::try_new(
            target.clone(),
            OffsetBuffer::new(offsets.iter().map(|o| *o as i32).collect()),
            values.as_struct().clone(),
            nulls,
            *sorted,
        )?),
        _ => unreachable!("matched above"),
    })
}

/// The Arrow C++ type error of a map cast to a list of other than a struct
/// of two fields.
pub(crate) fn map_entries(to_type: &DataType) -> ArrowError {
    ArrowError::SchemaError(format!(
        "Map type must be cast to a list<struct> with exactly two fields, not {to_type}"
    ))
}

/// Casts a list, large list, list view or large list view array to a fixed
/// size list as Arrow C++ `CastVarToFixedList` does: every list that is not
/// null must hold `size` values, every value of the child converts before
/// any is selected, and the lists then select their values.
///
/// # Errors
///
/// [`ArrowError::CastError`] (C++ `Invalid`) for a list of another size, and
/// any error of converting the values.
pub(crate) fn conversion_list_to_fixed_size_list(
    array: &dyn Array,
    field: &FieldRef,
    size: i32,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let CppList {
        offsets, values, ..
    } = CppList::new(array);
    let width = size as i64;
    let mut expected = offsets[0] + width;
    let mut nulls_have_correct_length = true;
    for (i, &end) in offsets.iter().enumerate().skip(1) {
        if array.is_null(i - 1) {
            nulls_have_correct_length &= end == expected;
            expected = end + width;
        } else if end != expected {
            return Err(ArrowError::CastError(
                "ListType can only be casted to FixedSizeListType if the lists are all the \
                 expected size."
                    .to_string(),
            ));
        } else {
            expected += width;
        }
    }
    let values = cast_field_with_context(&values, item(array.data_type()), field, cast_options)?;
    let len = array.len() * size as usize;
    let values = if array.null_count() > 0 && !nulls_have_correct_length {
        let indices: UInt64Array = (0..array.len())
            .flat_map(|i| {
                let start = (!array.is_null(i)).then(|| offsets[i] as u64);
                (0..size as u64).map(move |j| start.map(|start| start + j))
            })
            .collect();
        take(&values, &indices, None)?
    } else {
        let start = offsets[0] as usize;
        if start + len > values.len() {
            return Err(ArrowError::CastError(format!(
                "The offsets of the {} array do not delimit lists of its values",
                array.data_type()
            )));
        }
        values.slice(start, len)
    };
    Ok(Arc::new(FixedSizeListArray::try_new_with_length(
        field.clone(),
        size,
        values,
        array.nulls().cloned(),
        array.len(),
    )?))
}
