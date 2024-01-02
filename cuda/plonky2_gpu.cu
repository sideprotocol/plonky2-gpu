
#include "plonky2_gpu_impl.cuh"

struct CudaInvContext {
    cudaStream_t stream;
    cudaStream_t stream2;
};
template <class T>
struct DataSlice{
    T* ptr;
    int len;
};


#ifndef __CUDA_ARCH__
#include <string>
#include <mutex>

struct RustError { /* to be returned exclusively by value */
    int code;
    char *message;
#ifdef __cplusplus
    RustError(int e = 0) : code(e)
    {   message = nullptr;   }
    RustError(int e, const std::string& str) : code(e)
    {   message = str.empty() ? nullptr : strdup(str.c_str());   }
    RustError(int e, const char *str) : code(e)
    {   message = str==nullptr ? nullptr : strdup(str);   }
    // no destructor[!], Rust takes care of the |message|
#endif
};

#define CUDA_ASSERT(expr) \
do {\
    if (auto code = expr; code != cudaSuccess) {\
        printf("%s@%d failed: %s\n", #expr, __LINE__, cudaGetErrorString(code));\
        return RustError{code};\
    }\
} while(0)

//static std::mutex mtx;  // 互斥锁
//bool has_init = false;
//cudaStream_t stream;

//extern "C" RustError init();

//void try_init() {
//    std::lock_guard<std::mutex> lock(mtx);  // 加锁
//    if (has_init)
//        return ;
//    init();
//}
//
#include <fstream>
#include <vector>

extern "C" {

//    RustError init()
//    {
////        printf("in init\n");
//        has_init = true;
//        cudaSetDevice(0);
//        cudaDeviceReset();
//        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//        return RustError{cudaSuccess};
//    }


    RustError ifft(
            GoldilocksField* d_values_flatten,
            int poly_num, int values_num_per_poly, int log_len,
            const GoldilocksField* d_root_table,
            GoldilocksField* p_inv,
            CudaInvContext* ctx
    ) {
        GoldilocksField n_inv = *p_inv;
        auto stream = ctx->stream;

        clock_t start = clock();
        ifft_kernel<<<poly_num, 32*8, 0, stream>>>(d_values_flatten, poly_num, values_num_per_poly, log_len, d_root_table, n_inv);
        cudaStreamSynchronize(stream);
        printf("ifft_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        return RustError{cudaSuccess};
    }

    RustError fft_blinding(
            GoldilocksField* d_values_flatten,
            GoldilocksField* d_ext_values_flatten,
            int poly_num, int values_num_per_poly, int log_len,
            const GoldilocksField* d_root_table2, const GoldilocksField* d_shift_powers,
            int rate_bits,
            int pad_extvalues_len,
            CudaInvContext* ctx
    ) {
        auto stream = ctx->stream;

        int thcnt = 0;
        int nthreads = 32;

        d_ext_values_flatten += pad_extvalues_len;


        clock_t start = clock();
        thcnt = values_num_per_poly * poly_num;
        nthreads = 32;
        lde_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_values_flatten, d_ext_values_flatten,
                                                                               poly_num, values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("lde_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly * poly_num;
        nthreads = 32;
        init_lde_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num,
                                                                                    values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("init_lde_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly * poly_num;
        nthreads = 32;
        mul_shift_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num,
                                                                                     values_num_per_poly, rate_bits,
                                                                                     d_shift_powers);
        cudaStreamSynchronize(stream);
        printf("mul_shift_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        fft_kernel<<<poly_num, 32 * 8, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly * (1 << rate_bits),
                                                    log_len + rate_bits, d_root_table2, rate_bits);
        cudaStreamSynchronize(stream);
        printf("fft_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);
    }


    RustError build_merkle_tree(
        GoldilocksField* d_ext_values_flatten,
        int poly_num, int values_num_per_poly, int log_len,
        int rate_bits, int salt_size,
        int cap_height,
        int pad_extvalues_len,
        CudaInvContext* ctx
    ) {
        int values_num_per_extpoly = values_num_per_poly * (1 << rate_bits);
        auto stream = ctx->stream;
        int ext_poly_num = poly_num + salt_size;

        int len_cap = 1 << cap_height;
        int num_digests = 2 * (values_num_per_extpoly - len_cap);

        int thcnt = 0;
        int nthreads = 32;
        d_ext_values_flatten += pad_extvalues_len;


        clock_t start = clock();
        thcnt = values_num_per_extpoly * poly_num;
        nthreads = 32;
        reverse_index_bits_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_ext_values_flatten,
                                                                                              poly_num,
                                                                                              values_num_per_extpoly,
                                                                                              log_len + rate_bits);
        cudaStreamSynchronize(stream);
        printf("reverse_index_bits_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);

        int log2_leaves_len = log_len + rate_bits;
        assert(cap_height <= log2_leaves_len);

        auto *d_digest_buf = (PoseidonHasher::HashOut *) (d_ext_values_flatten + values_num_per_extpoly * ext_poly_num);

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        hash_leaves_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
                d_ext_values_flatten, poly_num + salt_size, values_num_per_extpoly, d_digest_buf, len_cap, num_digests);
        cudaStreamSynchronize(stream);
        printf("hash_leaves_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        nthreads = 32 * 8;
        thcnt = len_cap * nthreads;
        reduce_digests_kernel<<<(thcnt + nthreads - 1) / nthreads, nthreads, 0, stream>>>(values_num_per_extpoly,
                                                                                          d_digest_buf, len_cap,
                                                                                          num_digests);
        cudaStreamSynchronize(stream);
        printf("reduce_digests_kernel elapsed: %.2lf\n", (double) (clock() - start) / CLOCKS_PER_SEC * 1000);
    }


    RustError transpose(
            GoldilocksField* d_ext_values_flatten,
            int poly_num, int values_num_per_poly,
            int rate_bits, int salt_size,
            int pad_extvalues_len,
            CudaInvContext* ctx
    ){
        int values_num_per_extpoly = values_num_per_poly*(1<<rate_bits);
        auto stream = ctx->stream;
        int ext_poly_num = poly_num + salt_size;

        int thcnt = 0;
        int nthreads = 32;
        d_ext_values_flatten += pad_extvalues_len;


        clock_t start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        transpose_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, d_ext_values_flatten - pad_extvalues_len, ext_poly_num, values_num_per_extpoly);
        cudaStreamSynchronize(stream);
        printf("transpose_kernel elapsed: %.2lf\n",  (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        return RustError{cudaSuccess};
    }

    RustError merkle_tree_from_values(
            GoldilocksField* d_values_flatten,
            GoldilocksField* d_ext_values_flatten,
            int poly_num, int values_num_per_poly, int log_len,
           const GoldilocksField* d_root_table, const GoldilocksField* d_root_table2, const GoldilocksField* d_shift_powers,
           GoldilocksField* p_inv, int rate_bits, int salt_size,
           int cap_height,
           int pad_extvalues_len,
           CudaInvContext* ctx
    ){
        assert(0);
        printf("start merkle_tree_from_values: poly_num:%d, values_num_per_poly:%d, log_len:%d, n_inv:%lu\n",
               poly_num, values_num_per_poly, log_len, p_inv->data);
        printf("d_values_flatten: %p, d_ext_values_flatten: %p\n", d_values_flatten, d_ext_values_flatten);

        int values_num_per_extpoly = values_num_per_poly*(1<<rate_bits);
        GoldilocksField n_inv = *p_inv;
        auto stream = ctx->stream;
        int ext_poly_num = poly_num + salt_size;

        int len_cap = 1 << cap_height;
        int num_digests = 2 * (values_num_per_extpoly - len_cap);

        int thcnt = 0;
        int nthreads = 32;
        double ifft_kernel_use, lde_kernel_use, mul_shift_kernel_use, fft_kernel_use, reverse_index_bits_kernel_use,
                hash_leaves_kernel_use, reduce_digests_kernel_use, transpose_kernel_use;

        d_ext_values_flatten += pad_extvalues_len;




//        if (poly_num == 20) {
//            std::vector<GoldilocksField> values_flatten(values_num_per_poly*poly_num);
//            CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0], d_values_flatten,  values_num_per_poly*poly_num*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("zs_partial_products-gpu.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(values_flatten.data()), values_flatten.size() * sizeof(uint64_t));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//        }

        clock_t start = clock();
//        cudaMemsetAsync(d_ext_values_flatten, 8*values_num_per_poly*poly_num*(1<<rate_bits), 0, ctx->stream2);
        ifft_kernel<<<poly_num, 32*8, 0, stream>>>(d_values_flatten, poly_num, values_num_per_poly, log_len, d_root_table, n_inv);
        cudaStreamSynchronize(stream);
//        cudaStreamSynchronize(ctx->stream2);
        printf("ifft_kernel elapsed: %.2lf\n", ifft_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//        if (poly_num == 20) {
//            std::vector<GoldilocksField> values_flatten(values_num_per_poly*poly_num);
//            CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0], d_values_flatten,  values_num_per_poly*poly_num*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("values_flatten-gpu.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(values_flatten.data()), values_flatten.size() * sizeof(uint64_t));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//        }

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        lde_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_values_flatten, d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("lde_kernel elapsed: %.2lf\n", lde_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        init_lde_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("init_lde_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        mul_shift_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits, d_shift_powers);
        cudaStreamSynchronize(stream);
        printf("mul_shift_kernel elapsed: %.2lf\n", mul_shift_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//        if (poly_num == 20)
//        {
//            std::vector<GoldilocksField> data(values_num_per_poly+100);
//            CUDA_ASSERT(cudaMemcpyAsync(&data[0], d_ext_values_flatten,  data.size()*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//            for (int i = 0; i < data.size(); ++i) {
//                if (i < values_num_per_poly)
//                    printf("first i: %d, val:%016lX\n", i, data[i].data);
//                else
//                    printf("second i: %d, val:%016lX\n", i, data[i].data);
//            }
//
//        }
//

        start = clock();
        fft_kernel<<<poly_num, 32*8, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly*(1<<rate_bits), log_len+rate_bits, d_root_table2, rate_bits);
        cudaStreamSynchronize(stream);
        printf("fft_kernel elapsed: %.2lf\n", fft_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);


//        if (poly_num == 20)
//        {
//            std::vector<GoldilocksField> outs(values_num_per_extpoly*poly_num);
//            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_ext_values_flatten,  outs.size()*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("fft_kernel-gpu.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(outs.data()), outs.size() * sizeof(uint64_t));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//
//        }

        start = clock();
        thcnt = values_num_per_extpoly*poly_num;
        nthreads = 32;
        reverse_index_bits_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_extpoly, log_len+rate_bits);
        cudaStreamSynchronize(stream);
        printf("reverse_index_bits_kernel elapsed: %.2lf\n", reverse_index_bits_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        int log2_leaves_len = log_len + rate_bits;
        assert(cap_height <= log2_leaves_len);

        auto *d_digest_buf = (PoseidonHasher::HashOut*)(d_ext_values_flatten + values_num_per_extpoly * ext_poly_num);

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        hash_leaves_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(
                d_ext_values_flatten, poly_num+salt_size, values_num_per_extpoly, d_digest_buf, len_cap, num_digests);
        cudaStreamSynchronize(stream);
        printf("hash_leaves_kernel elapsed: %.2lf\n", hash_leaves_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        nthreads = 32*8;
        thcnt = len_cap * nthreads;
        reduce_digests_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(values_num_per_extpoly, d_digest_buf, len_cap, num_digests);
        cudaStreamSynchronize(stream);
        printf("reduce_digests_kernel elapsed: %.2lf\n", reduce_digests_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//    printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

//    std::vector<PoseidonHasher::HashOut> outs(num_digests_and_caps);
//    CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_digest_buf,  num_digests_and_caps*sizeof(PoseidonHasher::HashOut), cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);
////    for (int i = 0; i < outs.size(); ++i) {
////        printf("idx: %d, ", i);
////        PRINT_HEX("hash", outs[i]);
////    }
//
//    PoseidonHasher::HashOut* cap_buf = &outs[num_digests];
//
//    for (int i = 0; i < len_cap; ++i) {
//        printf("cap idx: %d, ", i);
//        PRINT_HEX("hash", cap_buf[i]);
//    }

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        transpose_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, d_ext_values_flatten - pad_extvalues_len, ext_poly_num, values_num_per_extpoly);
        cudaStreamSynchronize(stream);
        printf("transpose_kernel elapsed: %.2lf\n", transpose_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        double total_use =
                ifft_kernel_use+
                lde_kernel_use+
                mul_shift_kernel_use+
                fft_kernel_use+
                reverse_index_bits_kernel_use+
                hash_leaves_kernel_use+
                reduce_digests_kernel_use+
                transpose_kernel_use;


//        if (poly_num == 20)
//        {
//            std::vector<GoldilocksField> outs(values_num_per_extpoly*poly_num);
//            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_ext_values_flatten - pad_extvalues_len,  outs.size()*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("partial_products-gpu.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(outs.data()), outs.size() * sizeof(uint64_t));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//
//        }
//
        printf("total use:%.2lf\n", total_use);
        return RustError{cudaSuccess};
    }

    RustError merkle_tree_from_coeffs(
            GoldilocksField* d_values_flatten,
            GoldilocksField* d_ext_values_flatten,
            int poly_num, int values_num_per_poly, int log_len,
            const GoldilocksField* d_root_table, const GoldilocksField* d_root_table2, const GoldilocksField* d_shift_powers,
            int rate_bits, int salt_size,
            int cap_height,
            int pad_extvalues_len,
            CudaInvContext* ctx
    ){
        printf("start merkle_tree_from_coeffs: poly_num:%d, values_num_per_poly:%d, log_len:%d\n",
               poly_num, values_num_per_poly, log_len);
        printf("d_values_flatten: %p, d_ext_values_flatten: %p\n", d_values_flatten, d_ext_values_flatten);

        int values_num_per_extpoly = values_num_per_poly*(1<<rate_bits);
        auto stream = ctx->stream;
        int ext_poly_num = poly_num + salt_size;

        int len_cap = 1 << cap_height;
        int num_digests = 2 * (values_num_per_extpoly - len_cap);

        int thcnt = 0;
        int nthreads = 32;
        double lde_kernel_use, mul_shift_kernel_use, fft_kernel_use, reverse_index_bits_kernel_use,
                hash_leaves_kernel_use, reduce_digests_kernel_use, transpose_kernel_use;

        d_ext_values_flatten += pad_extvalues_len;


        clock_t start;
//        if (poly_num == 234) {
//            std::vector<GoldilocksField> values_flatten(values_num_per_poly*poly_num);
//            CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0], d_values_flatten,  values_num_per_poly*poly_num*sizeof(GoldilocksField),
//                                        cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("values_flatten-gpu.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(values_flatten.data()), values_flatten.size() * sizeof(uint64_t));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//        }

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        lde_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_values_flatten, d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("lde_kernel elapsed: %.2lf\n", lde_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        init_lde_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits);
        cudaStreamSynchronize(stream);
        printf("init_lde_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        thcnt = values_num_per_poly*poly_num;
        nthreads = 32;
        mul_shift_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly, rate_bits, d_shift_powers);
        cudaStreamSynchronize(stream);
        printf("mul_shift_kernel elapsed: %.2lf\n", mul_shift_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

    //        if (poly_num == 20)
    //        {
    //            std::vector<GoldilocksField> data(values_num_per_poly+100);
    //            CUDA_ASSERT(cudaMemcpyAsync(&data[0], d_ext_values_flatten,  data.size()*sizeof(GoldilocksField),
    //                                        cudaMemcpyDeviceToHost, stream));
    //            cudaStreamSynchronize(stream);
    //            for (int i = 0; i < data.size(); ++i) {
    //                if (i < values_num_per_poly)
    //                    printf("first i: %d, val:%016lX\n", i, data[i].data);
    //                else
    //                    printf("second i: %d, val:%016lX\n", i, data[i].data);
    //            }
    //
    //        }
    //

        start = clock();
        fft_kernel<<<poly_num, 32*8, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly*(1<<rate_bits), log_len+rate_bits, d_root_table2, rate_bits);
        cudaStreamSynchronize(stream);
        printf("fft_kernel elapsed: %.2lf\n", fft_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);


    //        if (poly_num == 20)
    //        {
    //            std::vector<GoldilocksField> outs(values_num_per_extpoly*poly_num);
    //            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_ext_values_flatten,  outs.size()*sizeof(GoldilocksField),
    //                                        cudaMemcpyDeviceToHost, stream));
    //            cudaStreamSynchronize(stream);
    //
    //            std::ofstream file("fft_kernel-gpu.bin", std::ios::binary);
    //            if (file.is_open()) {
    //                file.write(reinterpret_cast<const char*>(outs.data()), outs.size() * sizeof(uint64_t));
    //                file.close();
    //                std::cout << "Data written to file." << std::endl;
    //            } else {
    //                std::cerr << "Failed to open file." << std::endl;
    //            }
    //
    //        }

        start = clock();
        thcnt = values_num_per_extpoly*poly_num;
        nthreads = 32;
        reverse_index_bits_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_extpoly, log_len+rate_bits);
        cudaStreamSynchronize(stream);
        printf("reverse_index_bits_kernel elapsed: %.2lf\n", reverse_index_bits_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        int log2_leaves_len = log_len + rate_bits;
        assert(cap_height <= log2_leaves_len);

        auto *d_digest_buf = (PoseidonHasher::HashOut*)(d_ext_values_flatten + values_num_per_extpoly * ext_poly_num);

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        hash_leaves_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(
                d_ext_values_flatten, poly_num+salt_size, values_num_per_extpoly, d_digest_buf, len_cap, num_digests);
        cudaStreamSynchronize(stream);
        printf("hash_leaves_kernel elapsed: %.2lf\n", hash_leaves_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        nthreads = 32*8;
        thcnt = len_cap * nthreads;
        reduce_digests_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(values_num_per_extpoly, d_digest_buf, len_cap, num_digests);
        cudaStreamSynchronize(stream);
        printf("reduce_digests_kernel elapsed: %.2lf\n", reduce_digests_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

    //    printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

    //    std::vector<PoseidonHasher::HashOut> outs(num_digests_and_caps);
    //    CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_digest_buf,  num_digests_and_caps*sizeof(PoseidonHasher::HashOut), cudaMemcpyDeviceToHost, stream));
    //    cudaStreamSynchronize(stream);
    ////    for (int i = 0; i < outs.size(); ++i) {
    ////        printf("idx: %d, ", i);
    ////        PRINT_HEX("hash", outs[i]);
    ////    }
    //
    //    PoseidonHasher::HashOut* cap_buf = &outs[num_digests];
    //
    //    for (int i = 0; i < len_cap; ++i) {
    //        printf("cap idx: %d, ", i);
    //        PRINT_HEX("hash", cap_buf[i]);
    //    }

        cudaStreamSynchronize(ctx->stream2);

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        transpose_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_ext_values_flatten, d_ext_values_flatten - pad_extvalues_len, ext_poly_num, values_num_per_extpoly);
        cudaStreamSynchronize(stream);
        printf("transpose_kernel elapsed: %.2lf\n", transpose_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        double total_use =
                lde_kernel_use+
                mul_shift_kernel_use+
                fft_kernel_use+
                reverse_index_bits_kernel_use+
                hash_leaves_kernel_use+
                reduce_digests_kernel_use+
                transpose_kernel_use;

        printf("total use:%.2lf\n", total_use);
        return RustError{cudaSuccess};
    }


    RustError compute_quotient_polys(
            GoldilocksField* d_ext_values_flatten,
            int poly_num, int values_num_per_poly, int log_len,
            const GoldilocksField* d_root_table2, const GoldilocksField* d_shift_inv_powers,
            int rate_bits, int salt_size,

            DataSlice<GoldilocksField>* zs_partial_products_commitment_leaves,
            DataSlice<GoldilocksField>* constants_sigmas_commitment_leaves,
            GoldilocksField *d_outs, GoldilocksField *d_quotient_polys,


            DataSlice<GoldilocksField>* points,
            DataSlice<GoldilocksField>* z_h_on_coset_evals,
            DataSlice<GoldilocksField>* z_h_on_coset_inverses,

            DataSlice<GoldilocksField>* k_is,
            DataSlice<GoldilocksField>* alphas,
            DataSlice<GoldilocksField>* betas,
            DataSlice<GoldilocksField>* gammas,

            CudaInvContext* ctx
    ) {

        int ext_poly_num = poly_num + salt_size;
        int values_num_per_extpoly = values_num_per_poly*(1<<rate_bits);
        auto stream = ctx->stream;
        int thcnt = 0;
        int nthreads = 32;
        clock_t start;

//        uint8_t *start_p = (uint8_t*)d_ext_values_flatten;
//        uint8_t *end_p   = (uint8_t*)(d_ext_values_flatten+values_num_per_extpoly*ext_poly_num);


//        auto k_is = read_fvec_to_dev("k_is.bin");
//        auto alphas = read_fvec_to_dev("alphas.bin");
//        auto betas = read_fvec_to_dev("betas.bin");
//        auto gammas = read_fvec_to_dev("gammas.bin");
//        auto points = read_fvec_to_dev("points.bin");
//        auto z_h_on_coset_evals = read_fvec_to_dev("z_h_on_coset.evals.bin");
//        auto z_h_on_coset_inverses = read_fvec_to_dev("z_h_on_coset.inverses.bin");

//        GoldilocksField *d_outs, *d_quotient_polys;
//
//        d_outs = (GoldilocksField*)start_p;
//        start_p += values_num_per_extpoly*2*sizeof(GoldilocksField);
//
//        d_quotient_polys = (GoldilocksField*)start_p;
//        start_p += values_num_per_extpoly*2*sizeof(GoldilocksField);
//
//        assert(start_p < end_p);

//        cudaStreamSynchronize(stream);
//        size_t total_dev_use = start_p-(uint8_t*)d_ext_values_flatten;
//        printf("total_dev_use: %fG\n", (double )total_dev_use/1024/1024/1024);

        int num_challenges = 2;
        int num_gate_constraints = 231;
        int num_constants = 8;
        int num_routed_wires = 80;
        int quotient_degree_factor = 8;
        int num_partial_products = 9;
        int constants_sigmas_commitment_leaf_len = 88;
        int zs_partial_products_commitment_leaf_len = 20;
        int wires_commitment_leaf_len = 234;

//    printf("%d, %d\n", constants_sigmas_commitment_leaves->len, values_num_per_extpoly*constants_sigmas_commitment_leaf_len);
        assert(constants_sigmas_commitment_leaves->len    == values_num_per_extpoly*constants_sigmas_commitment_leaf_len);
        assert(zs_partial_products_commitment_leaves->len == values_num_per_extpoly*zs_partial_products_commitment_leaf_len);
        assert(points->len == values_num_per_extpoly);
        assert(alphas->len == num_challenges);
        assert(betas->len == num_challenges);
        assert(gammas->len == num_challenges);

        start = clock();
        thcnt = 300000;
        nthreads = 32;
        PoseidonHasher::HashOut public_inputs_hash = {
                GoldilocksField{0x672c5e6c12ad3476}, GoldilocksField{0xca5c2e49acfad27e},
                GoldilocksField{0x296be18388d15f70}, GoldilocksField{0x66b42e146a70d96d}
        };
        compute_quotient_values_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(
                log_len, rate_bits,
                points->ptr,
                d_outs,
                public_inputs_hash,

                constants_sigmas_commitment_leaves->ptr,     constants_sigmas_commitment_leaf_len,
                zs_partial_products_commitment_leaves->ptr,  zs_partial_products_commitment_leaf_len,
                d_ext_values_flatten,                wires_commitment_leaf_len,
                num_constants, num_routed_wires,
                num_challenges,
                num_gate_constraints,

                quotient_degree_factor,
                num_partial_products,

                z_h_on_coset_evals->ptr,
                z_h_on_coset_inverses->ptr,

                k_is->ptr,
                alphas->ptr,
                betas->ptr,
                gammas->ptr
        );
        if (auto code = cudaGetLastError(); code != cudaSuccess) {
            printf("compute quotient error: %s\n", cudaGetErrorString(code));
        }

        cudaStreamSynchronize(stream);
        printf("compute_quotient_values_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//        {
//            std::vector<GoldilocksField> outs(num_challenges*values_num_per_extpoly);
//            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_outs,  outs.size()*sizeof(GoldilocksField), cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("quotient_values.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(outs.data()), outs.size() * sizeof(GoldilocksField));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//
////        printf("v1: %lx, v2: %lx\n", outs[2086137].data, outs[2086137 + values_num_per_extpoly].data);
//        }

        start = clock();
        thcnt = values_num_per_extpoly;
        nthreads = 32;
        transpose_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_outs, d_quotient_polys, values_num_per_extpoly, num_challenges);
        cudaStreamSynchronize(stream);
        printf("transpose_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

        start = clock();
        GoldilocksField n_inv_ext = {.data = 0xfffff7ff00000801ULL};
        ifft_kernel<<<num_challenges, 32*8, 0, stream>>>(d_quotient_polys, num_challenges, values_num_per_extpoly, log_len+rate_bits, d_root_table2, n_inv_ext);
        cudaStreamSynchronize(stream);
        printf("ifft_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//        {
//            std::vector<GoldilocksField> outs(num_challenges*values_num_per_extpoly);
//            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_quotient_polys,  outs.size()*sizeof(GoldilocksField), cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            printf("v1: %lx, v2: %lx\n", outs[2086137].data, outs[2086137 + values_num_per_extpoly].data);
//        }

        start = clock();
        thcnt = values_num_per_extpoly*num_challenges;
        nthreads = 32;
        mul_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(d_quotient_polys, num_challenges, values_num_per_extpoly, d_shift_inv_powers);
        cudaStreamSynchronize(stream);
        printf("mul_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

//        {
//            std::vector<GoldilocksField> outs(num_challenges*values_num_per_extpoly);
//            CUDA_ASSERT(cudaMemcpyAsync(&outs[0], d_quotient_polys,  outs.size()*sizeof(GoldilocksField), cudaMemcpyDeviceToHost, stream));
//            cudaStreamSynchronize(stream);
//
//            std::ofstream file("quotient_values2.bin", std::ios::binary);
//            if (file.is_open()) {
//                file.write(reinterpret_cast<const char*>(outs.data()), outs.size() * sizeof(GoldilocksField));
//                file.close();
//                std::cout << "Data written to file." << std::endl;
//            } else {
//                std::cerr << "Failed to open file." << std::endl;
//            }
//
////        printf("v1: %lx, v2: %lx\n", outs[2086137].data, outs[2086137 + values_num_per_extpoly].data);
//        }
        return RustError{cudaSuccess};
    }

}

#endif
