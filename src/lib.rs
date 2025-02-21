#![allow(clippy::missing_safety_doc)]
#![allow(rustdoc::bare_urls)]
// The following line contains a markdown reference link.
// This is necessary to override the link destination in the README.md file, so
// that when the README.md is rendered standalone (e.g. on Github) it points to
// the online version, and when rendered by rustdoc to the docs module rendered
// page.
//! [developer documentation]: docs
#![doc = include_str!("../README.md")]

pub use self::{
    compiler::compile,
    ffi::{module_to_object, object_to_shared_lib, OptLevel},
    runtime::FormattedItem,
    values::Value,
};

mod arch;
pub mod cache;
mod compiler;
pub mod context;
pub mod debug;
pub mod docs;
pub mod error;
pub mod execution_result;
pub mod executor;
mod ffi;
pub mod libfuncs;
pub mod metadata;
pub mod module;
pub mod runtime;
pub mod starknet;
pub mod starknet_stub;
pub mod types;
pub mod utils;
mod values;
