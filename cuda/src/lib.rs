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

extern crate core;

use std::ffi::{c_char, c_ulong, c_void};
mod cuda {
    #[repr(C)]
    pub struct Error {
        pub code: i32,
        str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
    }

    impl Drop for Error {
        fn drop(&mut self) {
            extern "C" {
                fn free(str: Option<core::ptr::NonNull<i8>>);
            }
            unsafe { free(self.str) };
            self.str = None;
        }
    }

    impl From<Error> for String {
        fn from(status: Error) -> Self {
            let c_str = if let Some(ptr) = status.str {
                unsafe { std::ffi::CStr::from_ptr(ptr.as_ptr()) }
            } else {
                extern "C" {
                    fn cudaGetErrorString(code: i32) -> *const i8;
                }
                unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(status.code)) }
            };
            String::from(c_str.to_str().unwrap_or("unintelligible"))
        }
    }
}

#[repr(C)]
pub struct DataSlice {
    pub ptr: *const c_void,
    pub len: i32,
}

extern "C" {
    pub fn init();

    pub fn ifft(
        values_flatten: *mut u64,
        poly_num: i32,
        values_num_per_poly: i32,
        log_len: i32,
        root_table: *const u64,
        n_inv: *const u64,
        ctx: *mut c_void,
    ) -> cuda::Error;

    pub fn build_merkle_tree(
        ext_values_flatten: *mut u64,
        poly_num: i32,
        values_num_per_poly: i32,
        log_len: i32,
        rate_bits: i32,
        salt_size: i32,
        cap_height: i32,
        pad_extvalues_len: i32,
        ctx: *mut c_void,
    ) -> cuda::Error;

    pub fn merkle_tree_from_values(
        values_flatten: *mut u64,
        ext_values_flatten: *mut u64,
        poly_num: i32,
        values_num_per_poly: i32,
        log_len: i32,
        root_table: *const u64,
        root_table2: *const u64,
        shift_powers: *const u64,
        n_inv: *const u64,
        rate_bits: i32,
        salt_size: i32,
        cap_height: i32,
        pad_extvalues_len: i32,
        ctx: *mut c_void,
    ) -> cuda::Error;

    pub fn merkle_tree_from_coeffs(
        values_flatten: *mut u64,
        ext_values_flatten: *mut u64,
        poly_num: i32,
        values_num_per_poly: i32,
        log_len: i32,
        root_table: *const u64,
        root_table2: *const u64,
        shift_powers: *const u64,
        rate_bits: i32,
        salt_size: i32,
        cap_height: i32,
        pad_extvalues_len: i32,
        ctx: *mut c_void,
    ) -> cuda::Error;


    pub fn compute_quotient_polys(
        ext_values_flatten: *const u64,
        poly_num: i32,
        values_num_per_poly: i32,
        log_len: i32,
        root_table2: *const u64,
        shift_inv_powers: *const u64,
        rate_bits: i32,
        salt_size: i32,

        zs_partial_products_commitment_leaves: *const DataSlice,
        constants_sigmas_commitment_leaves: *const DataSlice,
        d_outs: *mut c_void,
        d_quotient_polys: *mut c_void,


        points: *const DataSlice,
        z_h_on_coset_evals: *const DataSlice,
        z_h_on_coset_inverses: *const DataSlice,

        k_is: *const DataSlice,
        alphas: *const DataSlice,
        betas: *const DataSlice,
        gammas: *const DataSlice,

        ctx: *mut c_void,
    ) -> cuda::Error;

}
