// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use std::{env, path::PathBuf};

fn main() {

    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };

    if nvcc.is_ok() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true);
        nvcc.flag("-g");
        nvcc.flag("-O5");
        nvcc.flag("-arch=sm_75");
        nvcc.flag("-maxrregcount=255");
        // nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
        // nvcc.flag("-Xcompiler").flag("-Wno-subobject-linkage");
        nvcc.file("plonky2_gpu.cu").compile("plonky2_cuda");

        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
    } else {
        println!("nvcc must be in the path. Consider adding /usr/local/cuda/bin.");
        // panic!();
    }
}
