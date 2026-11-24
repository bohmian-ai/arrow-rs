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

/// Helper function that takes a map container and casts the inner datatype.
pub(crate) fn cast_map_values(
    from: &MapArray,
    to_data_type: &DataType,
    cast_options: &CastContext,
    to_ordered: bool,
) -> Result<ArrayRef, ArrowError> {
    let DataType::Map(entries_field, _) = to_data_type else {
        return Err(ArrowError::CastError(
            "Internal Error: to_data_type is not a map type.".to_string(),
        ));
    };

    let key_field = key_field(entries_field).ok_or(ArrowError::CastError(
        "map is missing key field".to_string(),
    ))?;
    let value_field = value_field(entries_field).ok_or(ArrowError::CastError(
        "map is missing value field".to_string(),
    ))?;

    let key_array = cast_with_context(from.keys(), key_field.data_type(), cast_options)?;
    let value_array = cast_with_context(from.values(), value_field.data_type(), cast_options)?;

    Ok(Arc::new(MapArray::try_new(
        entries_field.clone(),
        from.offsets().clone(),
        StructArray::try_new(
            Fields::from(vec![key_field, value_field]),
            vec![key_array, value_array],
            from.entries().nulls().cloned(),
        )?,
        from.nulls().cloned(),
        to_ordered,
    )?))
}

/// Casts `from` as Arrow C++ does, as the list of its entries. A list or
/// large list target must hold a struct of exactly two fields, which take the
/// key and the value by position, as C++ `CastMap` does; a fixed size list
/// target casts the entries as a list of structs.
///
/// # Errors
///
/// [`ArrowError::SchemaError`], the Arrow C++ type error, for a list item
/// that is not a struct of two fields, and any error of casting the entries.
pub(crate) fn conversion_map_to_list(
    from: &MapArray,
    to_type: &DataType,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let entries = from.entries();
    let entries = match to_type {
        DataType::List(item) | DataType::LargeList(item) => {
            let DataType::Struct(fields) = item.data_type() else {
                return Err(two_fields(to_type));
            };
            let [key, value] = fields.iter().collect::<Vec<_>>()[..] else {
                return Err(two_fields(to_type));
            };
            let keys = cast_with_context(from.keys(), key.data_type(), cast_options)?;
            let values = cast_with_context(from.values(), value.data_type(), cast_options)?;
            let entries = StructArray::try_new(fields.clone(), vec![keys, values], None)?;
            let list = ListArray::try_new(
                item.clone(),
                from.offsets().clone(),
                Arc::new(entries),
                from.nulls().cloned(),
            )?;
            return cast_with_context(&list, to_type, cast_options);
        }
        _ => entries.clone(),
    };
    let item = Field::new("entries", entries.data_type().clone(), false);
    let list = ListArray::try_new(
        Arc::new(item),
        from.offsets().clone(),
        Arc::new(entries),
        from.nulls().cloned(),
    )?;
    cast_with_context(&list, to_type, cast_options)
}

/// The Arrow C++ type error of a map cast to a list of other than a struct
/// of two fields.
fn two_fields(to_type: &DataType) -> ArrowError {
    ArrowError::SchemaError(format!(
        "Map type must be cast to a list<struct> with exactly two fields, not {to_type}"
    ))
}

/// Gets the key field from the entries of a map.  For all other types returns None.
pub(crate) fn key_field(entries_field: &FieldRef) -> Option<FieldRef> {
    if let DataType::Struct(fields) = entries_field.data_type() {
        fields.first().cloned()
    } else {
        None
    }
}

/// Gets the value field from the entries of a map.  For all other types returns None.
pub(crate) fn value_field(entries_field: &FieldRef) -> Option<FieldRef> {
    if let DataType::Struct(fields) = entries_field.data_type() {
        fields.get(1).cloned()
    } else {
        None
    }
}
