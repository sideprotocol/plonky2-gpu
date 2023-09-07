use alloc::format;
use alloc::vec::Vec;
use std::cmp::{max, min};
use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::mem;
use std::mem::transmute;
use std::ops::IndexMut;
use std::process::exit;
use std::sync::Arc;

use itertools::Itertools;
use maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::{field, timed};
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};
use plonky2_cuda;
use plonky2_field::packable::Packable;
use cudart;
use cudart::memory::CudaMutSlice;
use cudart::memory::CudaSlice;
// use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceSlice};
use rustacuda::prelude::*;
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceSlice};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

pub struct CudaInnerContext {
    // pub stream: cudart::stream::CudaStream,
    // pub stream2: cudart::stream::CudaStream,
    pub stream: rustacuda::stream::Stream,
    pub stream2: rustacuda::stream::Stream,

}

#[repr(C)]
// pub struct CudaInvContext<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
pub struct CudaInvContext<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub inner: CudaInnerContext,
    pub ext_values_flatten :Arc<Vec<F>>,
    pub values_flatten     :Arc<Vec<F>>,
    pub digests_and_caps_buf :Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub ext_values_flatten2 :Arc<Vec<F>>,
    pub values_flatten2     :Arc<Vec<F>>,
    pub digests_and_caps_buf2 :Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub values_device: DeviceBuffer::<F>,
    pub ext_values_device: DeviceBuffer::<F>,
    pub root_table_device: DeviceBuffer::<F>,
    pub root_table_device2: DeviceBuffer::<F>,
    pub shift_powers_device: DeviceBuffer::<F>,

    // pub values_device: cudart::memory::DeviceAllocation::<'a, F>,
    // pub ext_values_device: cudart::memory::DeviceAllocation::<'a, F>,
    // pub root_table_device: cudart::memory::DeviceAllocation::<'a, F>,
    // pub root_table_device2: cudart::memory::DeviceAllocation::<'a, F>,
    // pub shift_powers_device: cudart::memory::DeviceAllocation::<'a, F>,
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
    pub my_polynomials: Vec<PolynomialCoeffs<F>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    ///
    // pub fn from_values_with_gpu_old(
    //     values: Vec<PolynomialValues<F>>,
    //     rate_bits: usize,
    //     blinding: bool,
    //     cap_height: usize,
    //     timing: &mut TimingTree,
    //     fft_root_table: Option<&FftRootTable<F>>,
    //     fft_root_table_deg: &Vec<F>,
    //     ctx: &mut plonky2_cuda::CudaInvContext<'_, F>,
    // ) -> Self
    // {
    //     // let oldLen = values[0].values.len();
    //     // let values: Vec<_> = values.into_iter().map(|mut poly| {
    //     //     poly.values = poly.values.iter().cycle()
    //     //         .take(poly.values.len() * 8)
    //     //         .cloned()
    //     //         .collect();
    //     //     PolynomialValues{values:poly.values}
    //     // }).collect();
    //     // let fft_root_table_new = crate::field::fft::fft_root_table(1<<(log2_strict(oldLen)+3)).concat();
    //     // let fft_root_table_deg = &fft_root_table_new;
    //
    //     let poly_num = values.len();
    //     let values_num_per_poly  = values[0].values.len();
    //
    //     // println!("buf1: {}",
    //     //          unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&values[0].values) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
    //     // );
    //
    //     let mut values_flatten = timed!(
    //         timing,
    //         "flat map",
    //         values.into_iter().flat_map(|poly| poly.values).collect::<Vec<F>>()
    //     );
    //     // println!("buf2: {}",
    //     //          unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&values_flatten) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
    //     // );
    //
    //     println!("hello gpu");
    //     // unsafe {
    //     //     let mut file = File::create("values.bin").unwrap();
    //     //     for value in &values_flatten {
    //     //         // if i < 9 {
    //     //         //     println!("{:016X}, ", value);
    //     //         // }
    //     //         file.write_all(mem::transmute::<&F, &[u8; 8]>(value)).unwrap();
    //     //     }
    //     //     // println!();
    //     // }
    //     // unsafe {
    //     //     let mut file = File::create("roots.bin").unwrap();
    //     //     for value in fft_root_table_deg {
    //     //         file.write_all(mem::transmute::<&F, &[u8; 8]>(value)).unwrap();
    //     //     }
    //     // }
    //     // exit(0);
    //
    //     let lg_n = log2_strict(values_num_per_poly );
    //     let n_inv = F::inverse_2exp(lg_n);
    //     let n_inv_ptr  : *const F = &n_inv;
    //
    //     let mut values_device = timed!(timing, "alloc values", {
    //             let mut values_device = cudart::memory::DeviceAllocation::<F>::alloc(values_flatten.len() * (1<<rate_bits)).unwrap();
    //             cudart::memory::memory_copy(&mut values_device.index_mut(0..values_flatten.len()), &values_flatten).unwrap();
    //             values_device
    //         });
    //
    //     let root_table_device = timed!(timing, "alloc roots values", {
    //             let mut root_table_device = cudart::memory::DeviceAllocation::<F>::alloc(fft_root_table_deg.len()).unwrap();
    //             cudart::memory::memory_copy(&mut root_table_device.index_mut(0..fft_root_table_deg.len()), &fft_root_table_deg).unwrap();
    //             root_table_device
    //         });
    //
    //     let fft_root_table_max = fft_root_table.unwrap().concat();
    //     let root_table_device2 = timed!(timing, "alloc roots2 values", {
    //             let mut root_table_device = cudart::memory::DeviceAllocation::<F>::alloc(fft_root_table_max.len()).unwrap();
    //             cudart::memory::memory_copy(&mut root_table_device.index_mut(0..fft_root_table_max.len()), &fft_root_table_max).unwrap();
    //             root_table_device
    //         });
    //
    //     unsafe {
    //
    //         timed!(
    //             timing,
    //             "IFFT",
    //             {
    //                 plonky2_cuda::ifft(
    //                     values_device.as_mut_c_void_ptr() as *mut u64, poly_num as i32, values_num_per_poly as i32,
    //                     lg_n as i32,
    //                     root_table_device.as_c_void_ptr() as *const u64,
    //                     n_inv_ptr as *const u64,
    //                     ctx as *mut plonky2_cuda::CudaInvContext,
    //                 );
    //                 ctx.stream.synchronize().unwrap();
    //             }
    //         );
    //
    //
    //         // println!("res: {}",
    //         //          unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&values_flatten) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
    //         // );
    //         // let mut file = File::create("res-gpu.bin").unwrap();
    //         // for value in &values_flatten {
    //         //     // if i < 9 {
    //         //     //     println!("{:016X}, ", value);
    //         //     // }
    //         //     file.write_all(mem::transmute::<&F, &[u8; 8]>(value)).unwrap();
    //         // }
    //
    //     }
    //     timed!(
    //         timing,
    //         "copy result",
    //         {
    //             let alllen = values_flatten.len();
    //             cudart::memory::memory_copy_with_kind(&mut values_flatten, &values_device.index(0..alllen),
    //                 cudart::memory::CudaMemoryCopyKind::DeviceToHost).unwrap();
    //
    //         }
    //     );
    //
    //     let coeffs = values_flatten
    //         .chunks(values_num_per_poly ).map(|chunk|PolynomialCoeffs{coeffs: chunk.to_vec()}).collect::<Vec<_>>();
    //     // .chunks(values_num_per_poly ).map(|chunk|PolynomialCoeffs{coeffs: chunk[0..oldLen].to_vec()}).collect::<Vec<_>>();
    //
    //     {
    //         let polynomials = coeffs;
    //         let degree = polynomials[0].len();
    //         let lde_values = timed!(
    //             timing,
    //             "FFT + blinding",
    //             Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
    //         );
    //
    //         let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
    //         timed!(timing, "reverse index bits", reverse_index_bits_in_place(&mut leaves));
    //         let merkle_tree = timed!(
    //             timing,
    //             "build Merkle tree",
    //             MerkleTree::new(leaves, cap_height)
    //         );
    //
    //         Self {
    //             polynomials,
    //             merkle_tree,
    //             degree_log: log2_strict(degree),
    //             rate_bits,
    //             blinding,
    //         }
    //     }
    // }

    pub fn from_values_with_gpu(
        values: &Vec<F>,
        poly_num: usize,
        values_num_per_poly: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        fft_root_table_deg: &Vec<F>,
        ctx: &mut CudaInvContext<F, C, D>,
    ) -> Self
    {
        // let poly_num: usize = values.len();
        // let values_num_per_poly  = values[0].values.len();

        // let values_flatten = values;
        // let mut values_flatten = timed!(
        //     timing,
        //     "flat map",
        //     values.into_par_iter().flat_map(|poly| poly.values).collect::<Vec<F>>()
        // );

        let salt_size = if blinding { SALT_SIZE } else { 0 };
        println!("hello gpu");

        let lg_n = log2_strict(values_num_per_poly );
        let n_inv = F::inverse_2exp(lg_n);
        let n_inv_ptr  : *const F = &n_inv;

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (values_num_per_poly*(1<<rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        let values_flatten_len = poly_num*values_num_per_poly;
        let ext_values_flatten_len = (values_flatten_len+salt_size*values_num_per_poly) * (1<<rate_bits);
        let digests_and_caps_buf_len = num_digests_and_caps;

        let pad_extvalues_len = ext_values_flatten_len;


        let (ext_values_flatten, values_flatten, digests_and_caps_buf);

        if values_num_per_poly*poly_num == ctx.values_flatten.len() {
            println!("in first stage");
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten).unwrap();
            values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.values_flatten).unwrap();
            digests_and_caps_buf
                = Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(&mut ctx.digests_and_caps_buf).unwrap();
        } else {
        // } else if values_num_per_poly*poly_num == ctx.values_flatten2.len() {
            println!("in second stage");
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten2).unwrap();
            values_flatten     = Arc::<Vec<F>>::get_mut(&mut ctx.values_flatten2).unwrap();
            digests_and_caps_buf
                =  Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(&mut ctx.digests_and_caps_buf2).unwrap();
        }

        let values_device = &mut ctx.values_device;
        let ext_values_device = &mut ctx.ext_values_device;
        let root_table_device = &mut ctx.root_table_device;
        let root_table_device2 = &mut ctx.root_table_device2;
        let shift_powers_device = &mut ctx.shift_powers_device;
        // cudart::memory::memory_copy(&mut values_device.index_mut(0..values.len()), values).unwrap();
        // cudart::memory::memory_copy(&mut values_device.index_mut(0..values.len()), values).unwrap();

        // unsafe {
        //     DeviceSlice::<F>::async_copy_from(values_device, values, &ctx.inner.stream).unwrap();
        // }

        unsafe {
            transmute::<&mut DeviceBuffer<F>, &mut DeviceBuffer<u64>>(values_device).copy_from(
                transmute::<&Vec<F>, &Vec<u64>>(values),
                // &ctx.inner.stream
            ).unwrap();
            ctx.inner.stream.synchronize().unwrap();
        }

        unsafe {
            let ctx_ptr :*mut CudaInnerContext = &mut ctx.inner;
            timed!(
                timing,
                "FFT + build Merkle tree + transpose with gpu",
                {
                    plonky2_cuda::merkle_tree_from_values(
                        values_device.as_mut_ptr() as *mut u64,
                        ext_values_device.as_mut_ptr() as *mut u64,
                        poly_num as i32, values_num_per_poly as i32,
                        lg_n as i32,
                        root_table_device.as_ptr() as *const u64,
                        root_table_device2.as_ptr() as *const u64,
                        shift_powers_device.as_ptr() as *const u64,
                        n_inv_ptr as *const u64,
                        rate_bits as i32,
                        salt_size as i32,
                        cap_height as i32,
                        pad_extvalues_len as i32,
                        ctx_ptr as *mut core::ffi::c_void,
                    );
                }
            );
        }
        timed!(
            timing,
            "copy result",
            {
                let alllen = values_flatten_len;
                // cudart::memory::memory_copy_with_kind(values_flatten, &values_device.index(0..alllen),
                //     cudart::memory::CudaMemoryCopyKind::DeviceToHost).unwrap();

                unsafe {
                    transmute::<&DeviceBuffer<F>, &DeviceBuffer<u64>>(values_device).async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(values_flatten),
                    &ctx.inner.stream).unwrap();
                    ctx.inner.stream.synchronize().unwrap();
                }

                let mut alllen = ext_values_flatten_len;
                // cudart::memory::memory_copy_with_kind(ext_values_flatten, &ext_values_device.index(0..alllen),
                //     cudart::memory::CudaMemoryCopyKind::DeviceToHost).unwrap();
                assert!(ext_values_flatten.len() == ext_values_flatten_len);
                unsafe {
                    transmute::<&DeviceBuffer<F>, &DeviceBuffer<u64>>(ext_values_device).async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(ext_values_flatten),
                    &ctx.inner.stream).unwrap();
                    ctx.inner.stream.synchronize().unwrap();
                }


                alllen += pad_extvalues_len;

                let len_with_F = digests_and_caps_buf_len*4;
                let fs= unsafe { mem::transmute::<&mut Vec<_>, &mut Vec<F>>(digests_and_caps_buf) };

                unsafe {  fs.set_len(len_with_F);}
                println!("alllen: {}, digest_and_cap_buf_len: {}, diglen: {}", alllen, len_with_F, digests_and_caps_buf_len);
                // cudart::memory::memory_copy_with_kind(&mut *fs, &ext_values_device.index(alllen..alllen+len_with_F),
                //     cudart::memory::CudaMemoryCopyKind::DeviceToHost).unwrap();
                // ext_values_device[alllen..alllen+len_with_F].async_copy_to(fs, ctx.inner.stream).unwrap();
                unsafe {
                    transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(&ext_values_device[alllen..alllen+len_with_F]).async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(fs),
                    &ctx.inner.stream).unwrap();
                    ctx.inner.stream.synchronize().unwrap();
                }

                unsafe {  fs.set_len(len_with_F / 4);}
            }
        );

        let coeffs = values_flatten.par_chunks(values_num_per_poly).map(|chunk|PolynomialCoeffs{coeffs: chunk.to_vec()}).collect::<Vec<_>>();

        // let lde_values = ext_values_flatten
        //     .chunks(values_num_per_poly * (1 << rate_bits)).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

        // let leaves = timed!(timing, "build leaves",
        //     ext_values_flatten.par_chunks(poly_num+salt_size).map(|chunk| chunk.to_vec()).collect::<Vec<_>>());

        {
            let polynomials = coeffs;

            // let leaves = lde_values;
            // let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
            // timed!(timing, "reverse index bits", reverse_index_bits_in_place(&mut leaves));

            // let merkle_tree = timed!(
            //     timing,
            //     "build Merkle tree",
            //     MerkleTree::new2(leaves, cap_height)
            // );

            let (ctx_ext_values_flatten, ctx_digests_and_caps_buf);
            if values_num_per_poly*poly_num == ctx.values_flatten.len() {
                ctx_ext_values_flatten = ctx.ext_values_flatten.clone();
                ctx_digests_and_caps_buf = ctx.digests_and_caps_buf.clone();
            } else {
            // } else if values_num_per_poly*poly_num == ctx.values_flatten2.len() {
                ctx_ext_values_flatten   =  ctx.ext_values_flatten2.clone();
                ctx_digests_and_caps_buf =  ctx.digests_and_caps_buf2.clone();
            }

            let merkle_tree = MerkleTree {
                leaves: vec![],
                // leaves,
                // digests: digests_and_caps_buf[0..num_digests].to_vec(),
                digests: vec![],
                cap: MerkleCap(ctx_digests_and_caps_buf[num_digests..num_digests_and_caps].to_vec()),
                my_leaf_len: poly_num+salt_size,
                // my_leaves: Arc::new(vec![]),
                my_leaves: ctx_ext_values_flatten,
                my_digests: ctx_digests_and_caps_buf,

            };

            // for (idx, h) in merkle_tree.digests.iter().enumerate() {
            //     // println!("hash: {:?}",  unsafe{std::mem::transmute::<&_, &[u8;32]>(&res)})
            //     let hex_string: String = unsafe{std::mem::transmute::<&_, &[u8;32]>(h)}.iter().map(|byte| format!("{:02x}", byte)).collect();
            //     let result: String = hex_string.chars()
            //         .collect::<Vec<char>>()
            //         .chunks(16)
            //         .map(|chunk| chunk.iter().collect::<String>())
            //         .collect::<Vec<String>>()
            //         .join(", ");
            //     println!("idx: {}, hash: {}", idx, result);
            // }
            //
            // for (idx, h) in merkle_tree.cap.0.iter().enumerate() {
            //     // println!("hash: {:?}",  unsafe{std::mem::transmute::<&_, &[u8;32]>(&res)})
            //     let hex_string: String = unsafe{std::mem::transmute::<&_, &[u8;32]>(h)}.iter().map(|byte| format!("{:02x}", byte)).collect();
            //     let result: String = hex_string.chars()
            //         .collect::<Vec<char>>()
            //         .chunks(16)
            //         .map(|chunk| chunk.iter().collect::<String>())
            //         .collect::<Vec<String>>()
            //         .join(", ");
            //     println!("cap idx: {}, hash: {}", idx, result);
            // }

            Self {
                polynomials,
                merkle_tree,
                degree_log: lg_n,
                rate_bits,
                blinding,
                my_polynomials: vec![],
            }
        }
    }

    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    type P = <F as Packable>::Packing;

    pub fn from_values2(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        // let mut file = File::create("values.bin").unwrap();
        //
        // values.iter().for_each(|p| {
        //     for value in &p.values {
        //         // if i < 9 {
        //         //     println!("{:016X}, ", value);
        //         // }
        //         file.write_all(unsafe{mem::transmute::<&F, &[u8; 8]>(value)}).unwrap();
        //     }
        //
        // });


        if false
        {
            let poly = values[4].clone();
            let n = poly.len();
            let lg_n = log2_strict(n);
            let n_inv = F::inverse_2exp(lg_n);

            let PolynomialValues { values: mut buffer } = poly;
            println!("buf1: {}",
                     unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&buffer) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
            );

            reverse_index_bits_in_place(& mut buffer);
            println!("buf2: {}",
                     unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&buffer) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
            );

            {
                let root_table = crate::field::fft::fft_root_table(buffer.len());
                let mut values = &mut buffer;

                let lg_packed_width = log2_strict(Self::P::WIDTH); // 0 when P is a scalar.
                let packed_values = Self::P::pack_slice_mut(values);
                let packed_n = packed_values.len();
                debug_assert!(packed_n == 1 << (lg_n - lg_packed_width));
                println!("lg_packed_width: {}", lg_packed_width);
                // Want the below for loop to unroll, hence the need for a literal.
                // This loop will not run when P is a scalar.
                let r = 0;
                assert!(lg_packed_width <= 4);
                for lg_half_m in 0..4 {
                    if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
                        // Intuitively, we split values into m slices: subarr[0], ..., subarr[m - 1]. Each of
                        // those slices is split into two halves: subarr[j].left, subarr[j].right. We do
                        // (subarr[j].left[k], subarr[j].right[k])
                        //   := f(subarr[j].left[k], subarr[j].right[k], omega[k]),
                        // where f(u, v, omega) = (u + omega * v, u - omega * v).
                        let half_m = 1 << lg_half_m;

                        // Set omega to root_table[lg_half_m][0..half_m] but repeated.
                        let mut omega = Self::P::default();
                        for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                            *omega_j = root_table[lg_half_m][j % half_m];
                        }

                        for k in (0..packed_n).step_by(2) {
                            // We have two vectors and want to do math on pairs of adjacent elements (or for
                            // lg_half_m > 0, pairs of adjacent blocks of elements). .interleave does the
                            // appropriate shuffling and is its own inverse.
                            let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                            let t = omega * v;
                            (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
                        }
                    }
                }

                // We've already done the first lg_packed_width (if they were required) iterations.
                let s = max(r, lg_packed_width);

                for lg_half_m in s..lg_n {
                    let lg_m = lg_half_m + 1;
                    let m = 1 << lg_m; // Subarray size (in field elements).
                    let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
                    let half_packed_m = packed_m / 2;
                    debug_assert!(half_packed_m != 0);

                    // omega values for this iteration, as slice of vectors
                    let omega_table = Self::P::pack_slice(&root_table[lg_half_m][..]);
                    for k in (0..packed_n).step_by(packed_m) {
                        for j in 0..half_packed_m {
                            let omega = omega_table[j];
                            let t = omega * packed_values[k + half_packed_m + j];
                            let u = packed_values[k + j];
                            packed_values[k + j] = u + t;
                            packed_values[k + half_packed_m + j] = u - t;
                        }
                    }
                    print!("buf5 lg_half_m:{}: ", lg_half_m);
                    for i in 0..8 {
                        let res = unsafe{std::mem::transmute::<&_, &u64>(&packed_values[i])};
                        print!("{:016X}, ", res);
                    }
                    println!();

                }

                // fft_classic(buffer, 0, computed_root_table);

            }
            // field::fft::fft_dispatch(&mut buffer, None, None);

            println!("buf3: {}",
                     unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&buffer) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
            );


            // We reverse all values except the first, and divide each by n.
            buffer[0] *= n_inv;
            buffer[n / 2] *= n_inv;
            for i in 1..(n / 2) {
                let j = n - i;
                let coeffs_i = buffer[j] * n_inv;
                let coeffs_j = buffer[i] * n_inv;
                buffer[i] = coeffs_i;
                buffer[j] = coeffs_j;
            }
            println!("buf4: {}",
                     unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&buffer) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
            );

            exit(0);
        }

        // let mut file = File::create("res-cpu-bits.bin").unwrap();

        let coeffs = timed!(
            timing,
            "IFFT",
            // values.into_iter().map(|mut v| {
            //     reverse_index_bits_in_place(&mut v.values);
            //     v.values.iter().for_each(|value| {
            //         file.write_all(unsafe{mem::transmute::<&F, &[u8; 8]>(value)}).unwrap();
            //     });
            //     v.ifft()
            // }).collect::<Vec<_>>()
            values.into_par_iter().map(|v| plonky2_field::fft::ifft_with_options(v, None, None)).collect::<Vec<_>>()
        );

        // println!("buf4: {}",
        //          unsafe { std::mem::transmute::<&Vec<F>, &Vec<u64>>(&coeffs[0].coeffs) }.iter().take(8).map(|&u| format!("{:016X}", u)).collect::<Vec<String>>().join(", ")
        // );

        // let mut file = File::create("res-cpu.bin").unwrap();
        // coeffs.iter().for_each(|coeffs| {
        //     for value in &coeffs.coeffs {
        //         // if i < 9 {
        //         //     println!("{:016X}, ", value);
        //         // }
        //         file.write_all(unsafe{mem::transmute::<&F, &[u8; 8]>(value)}).unwrap();
        //     }
        //
        // });
        // exit(0);

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        timed!(timing, "reverse index bits", reverse_index_bits_in_place(&mut leaves));
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
            my_polynomials: vec![],
        }
    }

    fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = {
            if self.merkle_tree.my_leaves.is_empty() {
                self.merkle_tree.leaves[index].as_slice()
            } else {
                &self.merkle_tree.my_leaves[index*self.merkle_tree.my_leaf_len .. (index+1)*self.merkle_tree.my_leaf_len]
            }
        };
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let quotient = composition_poly.divide_by_linear(*point);
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }
        // Multiply the final polynomial by `X`, so that `final_poly` has the maximum degree for
        // which the LDT will pass. See github.com/mir-protocol/plonky2/pull/436 for details.
        final_poly.coeffs.insert(0, F::Extension::ZERO);

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );
        println!("lde_final_poly len:{}, lde_final_values len: {}", lde_final_poly.len(), lde_final_values.len());

        let fri_proof = timed!(
            timing,
            "compute fri proof",
            fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            timing,
        ));

        fri_proof
    }
}
