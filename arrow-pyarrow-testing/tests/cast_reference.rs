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

//! Compares [`cast_with_conversion_options`] with the Apache Arrow C++ 24.0.0
//! cast operation on identical inputs.
//!
//! `cast_reference/corpus.py` translates the C++ cast tests of
//! `cpp/src/arrow/compute/kernels/scalar_cast_test.cc` at Apache Arrow
//! 24.0.0 (commit `31b4b6c0a0a7e7c117312d285541a21446675ec6`), and
//! `cast_reference/harness.py` executes each case through PyArrow 24.0.0. The
//! input of every execution crosses the C Data Interface unchanged, so slice
//! offsets, unreferenced dictionary values and the children of null parents
//! reach both engines. Each execution must agree on success or refusal, the
//! status category of a refusal, and the type, length, values and validity of
//! a result.
//!
//! Requires a Python interpreter with `pyarrow==24.0.0`, as the other tests
//! of this crate do. Set `CAST_REFERENCE_REPORT` to a path to write every
//! execution's outcome there.

use std::fmt::Write as _;

use arrow_array::{Array, make_array};
use arrow_cast::{ConversionOptions, cast_with_conversion_options};
use arrow_data::ArrayData;
use arrow_pyarrow::FromPyArrow;
use arrow_schema::{ArrowError, DataType};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The Arrow C++ status category a Rust cast error corresponds to.
fn category(error: &ArrowError) -> String {
    match error {
        ArrowError::NotYetImplemented(_) => "NotImplemented".to_string(),
        ArrowError::CastError(_) => "Invalid".to_string(),
        other => format!("unmapped Rust error {other:?}"),
    }
}

fn conversion_options(flags: &Bound<PyDict>) -> PyResult<ConversionOptions> {
    let flag = |name: &str| -> PyResult<bool> {
        flags
            .get_item(name)?
            .expect("the harness passes every flag")
            .extract()
    };
    Ok(ConversionOptions {
        allow_int_overflow: flag("allow_int_overflow")?,
        allow_time_truncate: flag("allow_time_truncate")?,
        allow_time_overflow: flag("allow_time_overflow")?,
        allow_decimal_truncate: flag("allow_decimal_truncate")?,
        allow_float_truncate: flag("allow_float_truncate")?,
        allow_invalid_utf8: flag("allow_invalid_utf8")?,
    })
}

/// Why the Rust outcome differs from the reference outcome, if it does.
fn mismatch(
    rust: &Result<ArrayData, ArrowError>,
    kind: &str,
    reference: Option<&ArrayData>,
    message: &str,
) -> Option<String> {
    match (rust, reference) {
        (Ok(actual), Some(expected)) => {
            if actual.data_type() != expected.data_type() {
                Some(format!(
                    "type {} != reference {}",
                    actual.data_type(),
                    expected.data_type()
                ))
            } else if actual != expected {
                Some(format!(
                    "values {:?} != reference {:?}",
                    make_array(actual.clone()),
                    make_array(expected.clone())
                ))
            } else {
                None
            }
        }
        (Ok(actual), None) => Some(format!(
            "converted to {:?}, reference refused with {kind}: {message}",
            make_array(actual.clone())
        )),
        (Err(error), Some(_)) => Some(format!("refused with {error}, reference converted")),
        (Ok(_), None) if kind == "supported" => None,
        (Err(error), None) if kind == "supported" => (category(error) == "NotImplemented")
            .then(|| format!("refused as unsupported ({error})")),
        (Err(error), None) if category(error) != kind => Some(format!(
            "refused as {} ({error}), reference as {kind}: {message}",
            category(error)
        )),
        (Err(_), None) => None,
    }
}

#[test]
fn cast_matches_arrow_cpp_reference() {
    Python::initialize();
    let mut report = String::new();
    let mut failures = Vec::new();
    let mut executions = 0;
    Python::attach(|py| -> PyResult<()> {
        let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cast_reference");
        let path = py.import("sys")?.getattr("path")?;
        path.call_method1("insert", (0, directory))?;
        let harness = py.import("harness")?;

        let unmapped = harness.call_method0("unmapped_tests")?;
        let unmapped = unmapped.cast::<PyList>()?;
        for test in unmapped.iter() {
            failures.push(format!("{test}: C++ cast test has no corpus mapping"));
        }

        for execution in harness.call_method0("executions")?.try_iter()? {
            let execution = execution?;
            let (test, case, variant, input, to, flags, kind, result, message, translation): (
                String,
                String,
                String,
                Bound<PyAny>,
                Bound<PyAny>,
                Bound<PyDict>,
                String,
                Option<Bound<PyAny>>,
                String,
                Option<String>,
            ) = execution.extract()?;
            executions += 1;
            let input = ArrayData::from_pyarrow_bound(&input)?;
            let to = DataType::from_pyarrow_bound(&to)?;
            let options = conversion_options(&flags)?;
            let reference = result
                .map(|result| ArrayData::from_pyarrow_bound(&result))
                .transpose()?;
            let rust = cast_with_conversion_options(&make_array(input), &to, &options)
                .map(|array| array.to_data());
            let rust_outcome = match &rust {
                Ok(_) => "ok".to_string(),
                Err(error) => category(error),
            };
            let difference = mismatch(&rust, &kind, reference.as_ref(), &message);
            writeln!(
                report,
                "{case} [{variant}] to {to} with {options:?}: reference {kind}, rust {rust_outcome}{}",
                difference
                    .as_ref()
                    .map(|d| format!(", MISMATCH {d}"))
                    .unwrap_or_default()
            )
            .unwrap();
            if let Some(translation) = translation {
                failures.push(format!("{case}: {translation}"));
            }
            if let Some(difference) = difference {
                failures.push(format!("{case} [{variant}] {test} to {to}: {difference}"));
            }
        }
        Ok(())
    })
    .unwrap_or_else(|error| {
        Python::attach(|py| error.print(py));
        panic!("the reference harness failed: {error}")
    });

    if let Ok(path) = std::env::var("CAST_REFERENCE_REPORT") {
        std::fs::write(path, &report).expect("the report path is writable");
    }
    assert!(executions > 0, "the corpus executed no cases");
    assert!(
        failures.is_empty(),
        "{} of {executions} executions disagree with Arrow C++:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
