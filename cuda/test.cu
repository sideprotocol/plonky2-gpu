#include "plonky2_gpu_impl.cuh"

#include <vector>
#include <fstream>
#undef CUDA_ASSERT

#define CUDA_ASSERT(expr) \
do {\
    if (auto code = expr; code != cudaSuccess) {\
        printf("%s@%d failed: %s\n", #expr, __LINE__, cudaGetErrorString(code));\
        return -1;\
    }\
} while(0)

#include <chrono>
#include <iostream>

static inline int ceil(int v, int v2) {
    assert(v < v2);
    return (v2+v-1)/v * v;
}
__global__
void test()
{
//    GoldilocksField n_inv = {.data = 18446673700670423041ULL};
//    GoldilocksField v = {.data = 0x000000002EF7A1BC};
//    printf("inv: %016lX\n", n_inv.data);
//    printf("n  : %016lX\n", v.data);
//    printf("res: %016lX\n", (n_inv * v - n_inv).data);

    GoldilocksField data[8] = {
	{12057761340118092379ULL},
	{6921394802928742357ULL},
	{401572749463996457ULL},
	{8075242603528285606ULL},
	{16383556155787439553ULL},
	{18045582516498195573ULL},
	{7296969412159674050ULL},
	{8317318176954617326ULL}
    };

    GoldilocksField state[SPONGE_WIDTH] = {0};
    for (int k = 0; k < SPONGE_RATE; ++k)
        state[k] = data[k];
    PoseidonHasher::permute_poseidon(state);
    auto out =  *(PoseidonHasher::HashOut*)state;
    PRINT_HEX("hash", out);
}

template <class T>
std::vector<T> read_vec_from_bin(std::string filename) {
    std::vector<T> res;
    std::ifstream file(filename, std::ios::binary);

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 根据文件大小调整vector容量
    res.resize(fileSize / sizeof(T));

    // 从文件中读取数据到vector
    file.read(reinterpret_cast<char*>(res.data()), fileSize);
    return res;
};

//auto read_hvec_from_bin = [](std::string filename) -> std::vector<PoseidonHasher::HashOut>{
//    std::vector<PoseidonHasher::HashOut> res;
//    std::ifstream file(filename, std::ios::binary);
//
//    // 获取文件大小
//    file.seekg(0, std::ios::end);
//    std::streampos fileSize = file.tellg();
//    file.seekg(0, std::ios::beg);
//
//    // 根据文件大小调整vector容量
//    res.resize(fileSize / sizeof(PoseidonHasher::HashOut));
//
//    // 从文件中读取数据到vector
//    file.read(reinterpret_cast<char*>(res.data()), fileSize);
//    return res;
//}

template <class T>
struct DataSlice{
    T* ptr;
    int len;
};

int main()
{
    cudaStream_t stream;
    cudaSetDevice(0);
    cudaDeviceReset();
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//    test<<<1, 1, 0, stream>>>();
//    cudaStreamSynchronize(stream);
//
//    exit(0);

    int poly_num = 234, values_num_per_poly = 262144, log_len = 18;
    constexpr int rate_bits = 3;
    int values_num_per_extpoly = values_num_per_poly*(1<<rate_bits);
    int cap_height = 4;
    int len_cap = 1 << cap_height;
    int salt_size = 0;
    int num_digests = 2 * (values_num_per_extpoly - len_cap);
    int num_digests_and_caps = num_digests + len_cap;
    int thcnt = 0;
    int nthreads = 32;
    int ext_poly_num = poly_num + salt_size;

    double ifft_kernel_use, lde_kernel_use, mul_shift_kernel_use, fft_kernel_use, reverse_index_bits_kernel_use,
                hash_leaves_kernel_use, reduce_digests_kernel_use, transpose_kernel_use;

#define  read_fvec_from_bin read_vec_from_bin<GoldilocksField>
#define  read_hvec_from_bin read_vec_from_bin<PoseidonHasher::HashOut>

    auto values_flatten = read_fvec_from_bin("values.bin");
    auto root_table  = read_fvec_from_bin("roots.bin");
    auto root_table2 = read_fvec_from_bin("roots2.bin");
    auto shift_powers = read_fvec_from_bin("powers.bin");


//    auto originalVector = values_flatten;
//    values_flatten.insert(values_flatten.end(), originalVector.begin(), originalVector.end());
//    poly_num *= 2;

    GoldilocksField *d_values_flatten;
    CUDA_ASSERT(cudaMalloc(&d_values_flatten, values_num_per_poly*poly_num*sizeof(GoldilocksField)));

    CUDA_ASSERT(cudaMemcpyAsync(d_values_flatten, &values_flatten[0],  values_num_per_poly*poly_num*sizeof(GoldilocksField),
                                cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);

    int pad_extvalues_len = values_num_per_extpoly*ext_poly_num;
    GoldilocksField *d_ext_values_flatten;
    CUDA_ASSERT(cudaMalloc(&d_ext_values_flatten,
                            (pad_extvalues_len + values_num_per_poly*ext_poly_num*(1<<rate_bits) + num_digests_and_caps*4)*sizeof(GoldilocksField)));
    d_ext_values_flatten += pad_extvalues_len;

    GoldilocksField *d_root_table;
    CUDA_ASSERT(cudaMalloc(&d_root_table, (values_num_per_poly+1)*sizeof(GoldilocksField)));

    CUDA_ASSERT(cudaMemcpyAsync(d_root_table, &root_table[0],  (values_num_per_poly+1)*sizeof(GoldilocksField),
                                cudaMemcpyHostToDevice, stream));

    cudaStreamSynchronize(stream);

    GoldilocksField *d_root_table2;
    CUDA_ASSERT(cudaMalloc(&d_root_table2, (values_num_per_poly*(1<<rate_bits)+1) * sizeof(GoldilocksField)));

    CUDA_ASSERT(cudaMemcpyAsync(d_root_table2, &root_table2[0],  (values_num_per_poly*(1<<rate_bits)+1) * sizeof(GoldilocksField),
                                cudaMemcpyHostToDevice, stream));

    cudaStreamSynchronize(stream);

    GoldilocksField *d_shift_powers;
    CUDA_ASSERT(cudaMalloc(&d_shift_powers, values_num_per_poly * sizeof(GoldilocksField)));

    CUDA_ASSERT(cudaMemcpyAsync(d_shift_powers, &shift_powers[0],  values_num_per_poly * sizeof(GoldilocksField),
                                cudaMemcpyHostToDevice, stream));

    cudaStreamSynchronize(stream);

//    printf("buf0: ");
//    for (int i = (1<<20); i < 8+(1<<20); ++i) {
//        printf("%016lX, ", values_flatten[i].data);
//    }
//    printf("\n");

//    CudaInvContext ctx = {.stream = stream};
    GoldilocksField n_inv = {.data = 18446673700670423041ULL};

//    ifft(d_values_flatten, poly_num, values_num_per_poly, log_len, d_root_table, &n_inv, &ctx);

//    reverse_index_bits_kernel<<<poly_num, 32, 0, stream>>>(d_values_flatten, poly_num, values_num_per_poly, log_len);
//    cudaStreamSynchronize(stream);
//
//    {
//        CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0], d_values_flatten,  values_num_per_poly*poly_num*sizeof(GoldilocksField),
//                                    cudaMemcpyDeviceToHost, stream));
//        cudaStreamSynchronize(stream);
//
//        std::ofstream file("res-gpu-bits.bin", std::ios::binary);
//        if (file.is_open()) {
//            file.write(reinterpret_cast<const char*>(values_flatten.data()), values_flatten.size() * sizeof(uint64_t));
//            file.close();
//            std::cout << "Data written to file." << std::endl;
//        } else {
//            std::cerr << "Failed to open file." << std::endl;
//        }
//
//    }





    clock_t start = clock();
//        cudaMemsetAsync(d_ext_values_flatten, 8*values_num_per_poly*poly_num*(1<<rate_bits), 0, ctx->stream2);
    ifft_kernel<<<poly_num, 32*8, 0, stream>>>(d_values_flatten, poly_num, values_num_per_poly, log_len, d_root_table, n_inv);
    cudaStreamSynchronize(stream);
//        cudaStreamSynchronize(ctx->stream2);
    printf("ifft_kernel elapsed: %.2lf\n", ifft_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);

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

    start = clock();
    fft_kernel<<<poly_num, 32*8, 0, stream>>>(d_ext_values_flatten, poly_num, values_num_per_poly*(1<<rate_bits), log_len+rate_bits, d_root_table2, rate_bits);
    cudaStreamSynchronize(stream);
    printf("fft_kernel elapsed: %.2lf\n", fft_kernel_use=(double )(clock()-start) / CLOCKS_PER_SEC * 1000);


//    CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0], d_values_flatten,  values_num_per_poly*poly_num*sizeof(GoldilocksField),
//                                cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);
//
//    std::ofstream file("values_flatten-gpu.bin", std::ios::binary);
//    if (file.is_open()) {
//        file.write(reinterpret_cast<const char*>(values_flatten.data()), values_flatten.size() * sizeof(uint64_t));
//        file.close();
//        std::cout << "Data written to file." << std::endl;
//    } else {
//        std::cerr << "Failed to open file." << std::endl;
//    }


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

//    start = clock();
//    for (int i = 0 ; i < 100; ++i) {
//        CUDA_ASSERT(cudaMemcpyAsync(&values_flatten[0 + i*2048], d_values_flatten + i *2048*100,  2048, cudaMemcpyDeviceToHost, stream));
//        cudaStreamSynchronize(stream);
//    }
//    printf("test: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

    double total_use =
            ifft_kernel_use+
            lde_kernel_use+
            mul_shift_kernel_use+
            fft_kernel_use+
            reverse_index_bits_kernel_use+
            hash_leaves_kernel_use+
            reduce_digests_kernel_use+
            transpose_kernel_use;

    printf("total use:%.2lf\n", total_use);

    uint8_t *start_p = (uint8_t*)d_ext_values_flatten;
    uint8_t *end_p   = (uint8_t*)(d_ext_values_flatten+values_num_per_extpoly*ext_poly_num);
//    {
//        std::vector<GoldilocksField> data;
//        uint8_t *old_p;
//        DataSlice<std::remove_reference<decltype(data[0])>::type>{(decltype(&data[0])) old_p, (int)data.size()};
//    }
//    std::remove_reference<decltype(data2[0])>::type asdf;

#define DO_MEMCPY_TODEV(data) \
            int cpylen = data.size() * sizeof(data[0]);                  \
            CUDA_ASSERT(cudaMemcpyAsync(start_p, &data[0],  cpylen, cudaMemcpyHostToDevice, stream)); \
            uint8_t *old_p = start_p; \
            start_p += cpylen;\
            assert(start_p < end_p);                  \
            DataSlice<std::remove_reference<decltype(data[0])>::type>{(decltype(&data[0])) old_p, (int)data.size()};

#define  read_fvec_to_dev(fname) \
    ({                             \
        auto data = read_vec_from_bin<GoldilocksField>(fname); \
        DO_MEMCPY_TODEV(data)                             \
    })

#define  read_hvec_to_dev(fname) \
    ({                             \
        auto data = read_vec_from_bin<PoseidonHasher::HashOut>(fname); \
        DO_MEMCPY_TODEV(data)                             \
    })

    auto zs_partial_products_commitment_polynomials = read_fvec_to_dev("zs_partial_products_commitment.polynomials.bin");
    auto zs_partial_products_commitment_leaves      = read_fvec_to_dev("zs_partial_products_commitment.leaves.bin");
    auto zs_partial_products_commitment_digests     = read_hvec_to_dev("zs_partial_products_commitment.digests.bin");
    auto zs_partial_products_commitment_caps        = read_hvec_to_dev("zs_partial_products_commitment.caps.bin");

    auto constants_sigmas_commitment_polynomials    = read_fvec_to_dev("constants_sigmas_commitment.polynomials.bin");
    auto constants_sigmas_commitment_leaves         = read_fvec_to_dev("constants_sigmas_commitment.leaves.bin");
    auto constants_sigmas_commitment_digests        = read_hvec_to_dev("constants_sigmas_commitment.digests.bin");
    auto constants_sigmas_commitment_caps           = read_hvec_to_dev("constants_sigmas_commitment.caps.bin");

    auto k_is = read_fvec_to_dev("k_is.bin");
    auto alphas = read_fvec_to_dev("alphas.bin");
    auto betas = read_fvec_to_dev("betas.bin");
    auto gammas = read_fvec_to_dev("gammas.bin");
    auto points = read_fvec_to_dev("points.bin");
    auto z_h_on_coset_evals = read_fvec_to_dev("z_h_on_coset.evals.bin");
    auto z_h_on_coset_inverses = read_fvec_to_dev("z_h_on_coset.inverses.bin");

    GoldilocksField *d_outs;
    CUDA_ASSERT(cudaMalloc(&d_outs, values_num_per_extpoly*2*sizeof(GoldilocksField)));

    cudaStreamSynchronize(stream);
    size_t total_dev_use = start_p-(uint8_t*)d_ext_values_flatten;
    printf("total_dev_use: %fG\n", (double )total_dev_use/1024/1024/1024);

    int num_challenges = 2;
    int num_gate_constraints = 231;
    int num_constants = 8;
    int num_routed_wires = 80;
    int quotient_degree_factor = 8;
    int num_partial_products = 9;
    int constants_sigmas_commitment_leaf_len = 88;
    int zs_partial_products_commitment_leaf_len = 20;
    int wires_commitment_leaf_len = 234;

//    printf("%d, %d\n", constants_sigmas_commitment_leaves.len, values_num_per_extpoly*constants_sigmas_commitment_leaf_len);
    assert(constants_sigmas_commitment_leaves.len    == values_num_per_extpoly*constants_sigmas_commitment_leaf_len);
    assert(zs_partial_products_commitment_leaves.len == values_num_per_extpoly*zs_partial_products_commitment_leaf_len);
    assert(points.len == values_num_per_extpoly);
    assert(alphas.len == num_challenges);
    assert(betas.len == num_challenges);
    assert(gammas.len == num_challenges);

    start = clock();
    thcnt = 15000;
    nthreads = 32;
    printf("values_num_per_extpoly: %d, log_len: %d\n", values_num_per_extpoly, log_len);
    PoseidonHasher::HashOut public_inputs_hash = {
            GoldilocksField{0x672c5e6c12ad3476}, GoldilocksField{0xca5c2e49acfad27e},
            GoldilocksField{0x296be18388d15f70}, GoldilocksField{0x66b42e146a70d96d}
    };;
    compute_quotient_values_kernel<<<(thcnt+nthreads-1)/nthreads, nthreads, 0, stream>>>(
            log_len, rate_bits,
            points.ptr,
            d_outs,
            public_inputs_hash,

            constants_sigmas_commitment_leaves.ptr,     constants_sigmas_commitment_leaf_len,
            zs_partial_products_commitment_leaves.ptr,  zs_partial_products_commitment_leaf_len,
            d_ext_values_flatten - pad_extvalues_len,                wires_commitment_leaf_len,
            num_constants, num_routed_wires,
            num_challenges,
            num_gate_constraints,

            quotient_degree_factor,
            num_partial_products,

            z_h_on_coset_evals.ptr,
            z_h_on_coset_inverses.ptr,

            k_is.ptr,
            alphas.ptr,
            betas.ptr,
            gammas.ptr
    );
    cudaStreamSynchronize(stream);
    printf("compute_quotient_values_kernel elapsed: %.2lf\n", (double )(clock()-start) / CLOCKS_PER_SEC * 1000);

    return 0;
}
