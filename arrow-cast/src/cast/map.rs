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

    let entries = from.entries().fields();
    let key_array = cast_field_with_context(from.keys(), &entries[0], &key_field, cast_options)?;
    let value_array =
        cast_field_with_context(from.values(), &entries[1], &value_field, cast_options)?;

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

/// Casts `from` to a fixed size list as Arrow C++ does, as the list of its
/// entries cast to a list of structs.
///
/// # Errors
///
/// The errors of [`conversion_list_to_fixed_size_list`].
pub(crate) fn conversion_map_to_fixed_size_list(
    from: &MapArray,
    field: &FieldRef,
    size: i32,
    cast_options: &CastContext,
) -> Result<ArrayRef, ArrowError> {
    let entries = from.entries();
    let item = Field::new("entries", entries.data_type().clone(), false);
    let list = ListArray::try_new(
        Arc::new(item),
        from.offsets().clone(),
        Arc::new(entries.clone()),
        from.nulls().cloned(),
    )?;
    conversion_list_to_fixed_size_list(&list, field, size, cast_options)
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
