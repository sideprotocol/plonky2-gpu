#include "def.cuh"

__global__
void ifft_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len, const GoldilocksField* root_table, GoldilocksField n_inv);

//__global__
//void reverse_index_bits_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len);

//#ifdef __CUDA_ARCH__
#if 1

#include <cassert>

__device__ inline
unsigned int bitrev(unsigned int num, const int log_len) {
    unsigned int reversedNum = 0;

    for (int i = 0; i < log_len; ++i) {
        if ((num & (1 << i)) != 0) {
            reversedNum |= 1 << ((log_len - 1) - i);
        }
    }

    return reversedNum;
}

__device__
void reverse_index_bits(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len)
{
//    int thCnt = get_global_thcnt();
//    int gid = get_global_id();
//
//    for (unsigned i = gid; i < values_num_per_poly*poly_num; i += thCnt) {
//        unsigned idx = i % values_num_per_poly;
//        unsigned poly_idx = i / values_num_per_poly;
//
//        unsigned ridx = bitrev(idx, log_len);
//        GoldilocksField *values = values_flatten + values_num_per_poly*poly_idx;
//        assert(ridx < values_num_per_poly);
//        if (idx < ridx) {
//            auto tmp = values[idx];
//            values[idx] = values[ridx];
//            values[ridx] = tmp;
//        }
//
//    }

    int thCnt = get_global_thcnt();
    int gid = get_global_id();
//    if (thCnt >= values_num_per_poly * poly_num)
//        return;

    assert((1 << log_len) == values_num_per_poly);
    assert(thCnt >= poly_num);

    int perpoly_thcnt = thCnt / poly_num;
    int poly_idx     = gid / perpoly_thcnt;
    int value_idx    = gid % perpoly_thcnt;

    assert(poly_idx < poly_num);

    for (unsigned i = value_idx; i < values_num_per_poly; i += perpoly_thcnt) {
        unsigned idx = i;

        unsigned ridx = bitrev(idx, log_len);
        GoldilocksField *values = values_flatten + values_num_per_poly*poly_idx;
        assert(ridx < values_num_per_poly);
        if (idx < ridx) {
            auto tmp = values[idx];
            values[idx] = values[ridx];
            values[ridx] = tmp;
        }
//        if (poly_idx == 232 && idx == 28724 && log_len == 21) {
//            printf("gid: %d, idx: %u, ridx: %u, addr: %p, value:%016lx, rvalue:%016lx\n", gid, idx, ridx, &values[idx], values[idx].data, values[ridx].data);
//        }

    }

    __syncthreads();
}

__global__
void reverse_index_bits_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len) {
    reverse_index_bits(values_flatten, poly_num, values_num_per_poly, log_len);
}

__device__
void fft_dispatch(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len, const GoldilocksField* root_table, int r) {
//    if (get_global_id() == 0 && poly_num == 20) {
//        printf("buf1: \n");
//        for (int j = 0; j < 20; ++j) {
//            for (int i = 0; i < values_num_per_poly; ++i) {
//                printf("%016lX\n", values_flatten[i + j*values_num_per_poly].data);
//            }
//            printf("end: %d\n", j);
//        }
//        printf("\n");
//    }
    reverse_index_bits(values_flatten, poly_num, values_num_per_poly, log_len);
//    if (get_global_id() == 0 && poly_num == 2) printf("after  reverse v1: %lx\n", values_flatten[0].data);
//    if (get_global_id() == 0 && poly_num == 2) printf("after  reverse v2: %lx\n", values_flatten[1].data);

//    __syncthreads();
//    if (get_global_id() == 0) {
//        printf("buf2: ");
//        for (int i = (1<<20); i < 8+(1<<20); ++i) {
//            printf("%016lX, ", values_flatten[i].data);
//        }
//        printf("\n");
//    }

    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    assert((1 << log_len) == values_num_per_poly);
    assert(thCnt >= poly_num);

    int perpoly_thcnt = thCnt / poly_num;
    int poly_idx     = gid / perpoly_thcnt;
    int value_idx    = gid % perpoly_thcnt;

//    assert(perpoly_thcnt % 32 == 0);
    assert(poly_idx < poly_num);

    int lg_packed_width = 0;
    int packed_n = values_num_per_poly;

    GoldilocksField* packed_values = values_flatten + values_num_per_poly*poly_idx;

    if (r > 0) {
        // if r == 0 then this loop is a noop.
        uint64_t mask = ~((1 << r) - 1);
        for (int i = value_idx; i < values_num_per_poly; i += perpoly_thcnt) {
            if (i % (1<<r) > 0) {
                assert(packed_values[i].data == 0);

//                if (packed_values[i].data != 0) {
//                    printf("in gid: %d, vid: %d, poly_idx: %d, i: %d, data: %016lx\n", gid, value_idx, poly_idx, i, packed_values[i].data);
////                    assert(0);
//                }
            }
            packed_values[i] = packed_values[i & mask];
        }
        __syncthreads();
    }

    int lg_half_m = r;
    for (; lg_half_m < log_len; ++lg_half_m) {
        int lg_m = lg_half_m + 1;
        int m = 1 << lg_m; // Subarray size (in field elements).
        int packed_m = m >> lg_packed_width; // Subarray size (in vectors).
        int half_packed_m = packed_m / 2;
        assert(half_packed_m != 0);

        const GoldilocksField* omega_table = root_table + ((1<<lg_half_m) - 1);
        if (lg_half_m > 0)
            omega_table += 1;

//            for (int k = 0; k < packed_n;  k += packed_m) {
//        for (int k = value_idx*perbatch_valcnt; k < (value_idx+1)*perbatch_valcnt;  k += packed_m) {
//            for (int j = 0; j < half_packed_m; ++j ) {
//        perpoly_thcnt / half_packed_m

//        for (int k = 0; k < packed_n/packed_m;  ++k) {
        for (int k = value_idx; k < packed_n/2;  k += perpoly_thcnt) {
            int kk = (k*2 / packed_m) * packed_m;
            int j  = k*2%packed_m / 2;
            GoldilocksField omega = omega_table[j];
            GoldilocksField t = omega * packed_values[kk + half_packed_m + j];
            GoldilocksField u = packed_values[kk + j];
            packed_values[kk + j] = u + t;
            packed_values[kk + half_packed_m + j] = u - t;
//            if (lg_half_m == 0 && poly_num == 2 && poly_idx == 0 && k == 0)
//                printf("in round 0 k: %d v1: %lx, omega: %lx, t: %lx, tt: %lx, u: %lx, kk:%d, j:%d\n",
//                       lg_half_m, values_flatten[0].data, omega.data, t.data, packed_values[kk + half_packed_m + j].data, u.data, kk, j);

        }

//        if (get_global_id() == 0) {
//            printf("buf5 lg_half_m:%d: ", lg_half_m);
//            for (int i = (1<<20); i < 8+(1<<20); ++i) {
//                printf("%016lX, ", packed_values[i].data);
//            }
//            printf("\n");
//        }
        __syncthreads();

//        if (value_idx == 0 && poly_num == 2 && poly_idx == 0) printf("in round: %d v1: %lx\n", lg_half_m, values_flatten[0].data);

    }
//    reverse_index_bits(values_flatten, poly_num, values_num_per_poly, log_len);


//    if (get_global_id() == 0) {
//        printf("buf3: ");
//        for (int i = (1<<20); i < 8+(1<<20); ++i) {
//            printf("%016lX, ", values_flatten[i].data);
//        }
//        printf("\n");
//    }
//


//    if (get_global_id() == 0) {
//        printf("buf4: ");
//        for (int i = (1<<20); i < 8+(1<<20); ++i) {
//            printf("%016lX, ", values_flatten[i].data);
//        }
//        printf("\n");
//    }

}

__global__
void ifft_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len, const GoldilocksField* root_table, GoldilocksField n_inv) {
//    if (get_global_id() == 0 && poly_num == 2) printf("before fft_dispatch v1: %lx\n", values_flatten[0].data);
//    if (get_global_id() == 0 && poly_num == 2) printf("before fft_dispatch v2: %lx\n", values_flatten[1<<20].data);
    fft_dispatch(values_flatten, poly_num, values_num_per_poly, log_len, root_table, 0);
//    if (get_global_id() == 0 && poly_num == 2) printf("after fft_dispatch v1: %lx\n", values_flatten[0].data);
//    if (get_global_id() == 0 && poly_num == 2) printf("after fft_dispatch v2: %lx\n", values_flatten[1<<20].data);

    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    assert((1 << log_len) == values_num_per_poly);
    assert(thCnt > poly_num);

    int perpoly_thcnt = thCnt / poly_num;
    int poly_idx     = gid / perpoly_thcnt;
    int value_idx    = gid % perpoly_thcnt;

    assert(perpoly_thcnt % 32 == 0);
    assert(poly_idx < poly_num);

    GoldilocksField* buffer = values_flatten + values_num_per_poly*poly_idx;

    if (value_idx == 0) {
        buffer[0] *= n_inv;
        buffer[values_num_per_poly / 2] *= n_inv;
    }

    assert(perpoly_thcnt < values_num_per_poly);
    for (int i = value_idx; i < values_num_per_poly/2; i += perpoly_thcnt) {
        if (i == 0)
            continue;
        int j = values_num_per_poly - i;
        GoldilocksField coeffs_i = buffer[j] * n_inv;
        GoldilocksField coeffs_j = buffer[i] * n_inv;
        buffer[i] = coeffs_i;
        buffer[j] = coeffs_j;
    }
}

__global__
void fft_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int log_len, const GoldilocksField* root_table, int r) {
    fft_dispatch(values_flatten, poly_num, values_num_per_poly, log_len, root_table, r);
}


__global__
void lde_kernel(const GoldilocksField* values_flatten, GoldilocksField* ext_values_flatten, int poly_num, int values_num_per_poly, int rate_bits)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    int values_num_per_poly2 = values_num_per_poly * (1<<rate_bits);
//    for (int i = gid; i < poly_num*values_num_per_poly2; i += thCnt) {
//        assert(ext_values_flatten[i].data == 0);
//    }
//    return;

    for (int i = gid; i < poly_num*values_num_per_poly; i += thCnt) {
        unsigned idx = i % values_num_per_poly;
        unsigned poly_idx = i / values_num_per_poly;
        assert(poly_idx < poly_num);
        const GoldilocksField *values = values_flatten + values_num_per_poly*poly_idx;
        ext_values_flatten[poly_idx*values_num_per_poly2 + idx] = values[idx];
    }
}

__global__
void init_lde_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int rate_bits)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    assert(thCnt > poly_num);

    int values_num_per_poly2 = values_num_per_poly * (1<<rate_bits);
    for (int i = gid; i < poly_num*values_num_per_poly*7; i += thCnt) {
        unsigned idx = i % (values_num_per_poly*7);
        unsigned poly_idx = i / (values_num_per_poly*7);

        GoldilocksField* values = values_flatten + poly_idx*values_num_per_poly2 + values_num_per_poly;
        values[idx].data = 0;
    }

}
__global__
void mul_shift_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, int rate_bits, const GoldilocksField* shift_powers)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    uint64_t values_num_per_poly2 = values_num_per_poly * (1<<rate_bits);
    for (int i = gid; i < poly_num*values_num_per_poly; i += thCnt) {
        unsigned idx = i % values_num_per_poly;
        unsigned poly_idx = i / values_num_per_poly;

        GoldilocksField* values = values_flatten + poly_idx*values_num_per_poly2;
        values[idx] *= shift_powers[idx];
    }
}

static __device__ inline int find_digest_index(int layer, int idx, int cap_len, int digest_len)
{
    int d_idx = 0;
    int d_len = digest_len;
    int c_len = cap_len;

    assert(idx < cap_len/(1<<layer));
    idx *= (1<<layer);

    bool at_right;
    while (c_len > (1<<layer)) {
        assert(d_len % 2 == 0);
        at_right = false;
        if (idx >= c_len / 2) {
            d_idx += d_len/2 +(d_len>2);
            idx -= c_len/2;
            at_right = true;
        }
        c_len = c_len/2;
        d_len = d_len/2 - 1;
    }

    if (layer > 0) {
//        idx += c_len;
//        c_len *= 2;
        d_len = 2*(d_len+1);
        if (at_right) {
            d_idx -= 1;
        } else
            d_idx += d_len/2 - 1;
    }
    assert(d_idx < digest_len && d_idx >= 0);
    return d_idx;
}
__global__
void hash_leaves_kernel(GoldilocksField* values_flatten, int poly_num, int leaves_len,
                        PoseidonHasher::HashOut* digest_buf, int len_cap, int num_digests)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    assert(num_digests % len_cap == 0);

//    int percap_digestnum = num_digests / len_cap;
    const int cap_len = leaves_len/len_cap;
    const int digest_len = num_digests/len_cap;
//    printf("cap_len: %d, digest_len:%d, leavse_len:%d \n", cap_len, digest_len, leaves_len);

    for (int i = gid; i < leaves_len; i += thCnt) {
        GoldilocksField state[SPONGE_WIDTH] = {0};

        for (int j = 0; j < poly_num; j += SPONGE_RATE) {
            for (int k = 0; k < SPONGE_RATE && (j+k)<poly_num; ++k)
                state[k] = *(values_flatten + leaves_len*(j+k) + i);
            PoseidonHasher::permute_poseidon(state);
        }

        const int ith_cap = i / cap_len;
        const int idx = i % cap_len;
        int d_idx = find_digest_index(0, idx, cap_len, digest_len);
//        if (i < 512)
//            printf("gid: %d, i:%d, ith_cap:%d, idx:%d, d_idx: %d\n", gid, i, ith_cap, idx, d_idx);

        assert((d_idx < digest_len));
        digest_buf[d_idx + ith_cap*digest_len] = *(PoseidonHasher::HashOut*)state;

//        if (ith_cap == 0 && idx >= cap_len/2)
//        {
////            for (int k = 0; k < 8; ++k)
////                printf("leaves%d: %lu\n", k, values_flatten[leaves_len*k].data);
////            printf("d_idx: %d \n", d_idx);
//
////            PRINT_HEX("hash", digest_buf[d_idx + ith_cap*digest_len]);
//            char buf[30 + sizeof(PoseidonHasher::HashOut)*2];
//            auto data = (uint8_t*)&digest_buf[d_idx + ith_cap*digest_len];
//            int k = 0;
//            int n = 0;
//            for (; k < sizeof(PoseidonHasher::HashOut); ++k) {
//                int v = data[k];
//                buf[k*2 +n]   = ((v >> 4)>9? (v >> 4)-10 +'a': (v >> 4) +'0');
//                buf[k*2+1 +n] = ((v & 0xF)>9? (v & 0xF)-10 +'a': (v & 0xF) +'0');
//
//                if ((k+1) % 8 == 0 && k != sizeof(PoseidonHasher::HashOut)-1) {
//                    buf[k*2+1+n +1] = ',';
//                    buf[k*2+1+n +2] = ' ';
//                    n+=2;
//                }
//            }
//            buf[k*2 +n] = 0;
//            printf("capid: %d, idx: %d, hash: %s\n", ith_cap, idx, buf);
//        }

//        digest_buf[i] = *(PoseidonHasher::HashOut*)state;
    }
}

__global__
void reduce_digests_kernel(int leaves_len, PoseidonHasher::HashOut* digest_buf, int len_cap, int num_digests) {
    int thCnt = get_global_thcnt();
    int gid = get_global_id();
    assert(num_digests % len_cap == 0);
    const int percap_thnum = thCnt / len_cap;
    assert(percap_thnum % 32 == 0);

    const int ith_cap = gid / percap_thnum;
    const int cap_idx = gid % percap_thnum;

    int cap_len = leaves_len/len_cap;
    const int digest_len = num_digests/len_cap;

    PoseidonHasher::HashOut* cap_buf = digest_buf + num_digests;
    digest_buf += digest_len * ith_cap;

    const int old_cap_len = cap_len;
    for (int layer = 0; cap_len > 1; ++layer, cap_len /= 2) {
        for (int i = cap_idx; i < cap_len/2; i += percap_thnum) {
            int idx1 = find_digest_index(layer, i*2,    old_cap_len, digest_len);
            int idx2 = find_digest_index(layer, i*2 +1, old_cap_len, digest_len);

            auto h1 = digest_buf[idx1];
            auto h2 = digest_buf[idx2];

            GoldilocksField perm_inputs[SPONGE_WIDTH] = {0};
            *((PoseidonHasher::HashOut*)&perm_inputs[0]) = h1;
            *((PoseidonHasher::HashOut*)&perm_inputs[4]) = h2;

            PoseidonHasher::permute_poseidon(perm_inputs);

            if (cap_len == 2) {
                assert(old_cap_len > (1<<layer));
                cap_buf[ith_cap] = *(PoseidonHasher::HashOut*)perm_inputs;

//                printf("cap: %d, ", ith_cap);
//                printf("h1 "); PRINT_HEX("hash", h1);
//                printf("h2 "); PRINT_HEX("hash", h2);
//
//                PRINT_HEX("hash", cap_buf[ith_cap]);
            } else {
                int idx3 = find_digest_index(layer+1, i, old_cap_len, digest_len);
//                if (ith_cap == 0 && i < 100 && layer == 0)
//                    printf("i: %d, idx: %d\n", i, idx3);
                digest_buf[idx3] = *(PoseidonHasher::HashOut*)perm_inputs;
            }
        }
        __syncthreads();
    }

}

__global__
void transpose_kernel(GoldilocksField* src_values_flatten, GoldilocksField* dst_values_flatten, int poly_num, int values_num_per_poly)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    for (int i = gid; i < poly_num*values_num_per_poly; i += thCnt) {
        unsigned val_idx = i / poly_num;
        unsigned poly_idx = i % poly_num;

        GoldilocksField *src_value = src_values_flatten + poly_idx * values_num_per_poly + val_idx;
        GoldilocksField *dst_value = dst_values_flatten + val_idx * poly_num + poly_idx;

        *dst_value = *src_value;
    }
}


#include "gates-def.cuh"


__global__
void compute_quotient_values_kernel(
        int degree_log, int rate_bits, GoldilocksField* points, GoldilocksField* outs,
        PoseidonHasher::HashOut public_inputs_hash,


        GoldilocksField* constants_sigmas_commitment_leaves,     int constants_sigmas_commitment_leaf_len,
        GoldilocksField* zs_partial_products_commitment_leaves,  int zs_partial_products_commitment_leaf_len,
        GoldilocksField* wires_commitment_leaves,                int wires_commitment_leaf_len,
        int num_constants, int _num_routed_wires,
        int _num_challenges,
        int _num_gate_constraints,

        int _quotient_degree_factor,
        int num_partial_products,

        GoldilocksField* z_h_on_coset_evals,
        GoldilocksField* z_h_on_coset_inverses,

        GoldilocksField* k_is,
        GoldilocksField* alphas,
        GoldilocksField* betas,
        GoldilocksField* gammas

)
{
    constexpr int num_challenges = 2;
    constexpr int num_gate_constraints = 231;
    assert(num_gate_constraints == _num_gate_constraints);
    assert(num_challenges == _num_challenges);

    int thCnt = get_global_thcnt();
    int gid = get_global_id();

//    if (gid == 0) {
//        auto res = GoldilocksField::from_canonical_u64(0xfff923c55a2e4a87) * GoldilocksField::from_canonical_u64(0xbfa99fe2edeb56f5);
//        printf("mul: %lx\n", res.data);
////        assert(res == GoldilocksField::from_canonical_u64(1));
//    }

    int step = 1;
    int next_step = 8;
    int values_num_per_extpoly = (1<<(rate_bits+degree_log));
//    int values_num_per_extpoly = 1;
    int lde_size  = values_num_per_extpoly;

    constexpr int quotient_degree_factor = 8;
    constexpr int num_routed_wires = 80;
    constexpr int max_degree = quotient_degree_factor;
    int num_prods = num_partial_products;

//    if (gid == 0) {
//        GoldilocksFieldView{alphas, num_challenges}.print_hex("alphas");
//        GoldilocksFieldView{betas, num_challenges}.print_hex("betas");
//        GoldilocksFieldView{gammas, num_challenges}.print_hex("gammas");
//
//    }

    auto get_lde_values = [degree_log, rate_bits](GoldilocksField* leaves, int leaf_len, int i, int step) -> GoldilocksFieldView {
        int index = i * step;
        index = bitrev(index, degree_log+rate_bits);
        return GoldilocksFieldView{&leaves[index*leaf_len], leaf_len};
    };

    for (int index = gid; index < values_num_per_extpoly; index += thCnt) {
        GoldilocksField x = GoldilocksField::coset_shift() * points[index];
        int i_next = (index + next_step) % lde_size;
        auto local_constants_sigmas = get_lde_values(constants_sigmas_commitment_leaves,
                                                     constants_sigmas_commitment_leaf_len, index, step);

        auto local_constants = local_constants_sigmas.view(0, num_constants);
        auto s_sigmas = local_constants_sigmas.view(num_constants, num_constants + num_routed_wires);
        auto local_wires = get_lde_values(wires_commitment_leaves, wires_commitment_leaf_len, index, step);
        auto local_zs_partial_products = get_lde_values(zs_partial_products_commitment_leaves,
                                                        zs_partial_products_commitment_leaf_len, index, step);
        auto local_zs = local_zs_partial_products.view(0, num_challenges);
        auto next_zs = get_lde_values(zs_partial_products_commitment_leaves, zs_partial_products_commitment_leaf_len,
                                      i_next, step).view(0, num_challenges);

        auto partial_products = local_zs_partial_products.view(num_challenges);

//        if (index == 1048576) {
//            printf("i: %d, len: %d, lcs: ", index, local_constants_sigmas.len);
//            local_constants_sigmas.print_hex();
//            printf("i: %d, len: %d, lw: ", index, local_wires.len);
//            local_wires.print_hex();
//            printf("i: %d, len: %d, lzpp: ", index, local_zs_partial_products.len);
//            local_zs_partial_products.print_hex();
//            printf("i: %d, len: %d, nzs: ", index, next_zs.len);
//            next_zs.print_hex();
//        }

//        let constraint_terms_batch =
//        evaluate_gate_constraints_base_batch::<F, C, D>(common_data, vars_batch);

        assert(num_routed_wires % max_degree == 0);

//        let constraint_terms = PackedStridedView::new(&constraint_terms_batch, n, k);

        GoldilocksField res[num_challenges] = {0};

        auto reduce_with_powers = [&res, &alphas, num_challenges](GoldilocksField term) {
            for (int i = 0; i < num_challenges; ++i) {
                res[i] = term + res[i] * alphas[i];
            }
        };

        GoldilocksField constraint_terms_batch[num_gate_constraints] = {0};
        auto evaluate_gate_constraints_base_batch = [&]()
        {
            struct SelectorsInfo {
                int *selector_indices;
                Range<int>* groups;
            };

            int selector_indices[25] = {
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5
            };

            constexpr  int num_selectors = 6;
            Range<int> groups[num_selectors] = {
                    Range<int>{0,6},
                    Range<int>{6,11},
                    Range<int>{11,16},
                    Range<int>{16,21},
                    Range<int>{21,24},
                    Range<int>{24,25}
            };
            SelectorsInfo selectors_info = {
                    .selector_indices = selector_indices,
                    .groups = groups
            };

            struct BaseGate {};
            using FUNC = void (BaseGate::*)(EvaluationVarsBasePacked, StridedConstraintConsumer yield_constr);

            struct GateFUNC {
                BaseGate* gate;
                FUNC func;
                int num_constraints;
            };
            constexpr const int num_gates = 25;
            GateFUNC gate_objs[num_gates];

#define DECL_GATE_NAME(TYPE, NAME, INDEX) \
        gate_objs[INDEX] = GateFUNC{.gate = (BaseGate*)&NAME, .func = (FUNC)&TYPE::eval_unfiltered_base_packed, .num_constraints = NAME.num_constraints()};

            NoopGate NoopGate_ins;
            DECL_GATE_NAME(NoopGate,NoopGate_ins, 0);
            ConstantGate ConstantGate_ins{ .num_consts = 2 };
            DECL_GATE_NAME(ConstantGate, ConstantGate_ins, 1);

            PublicInputGate PublicInputGate_ins;
            DECL_GATE_NAME(PublicInputGate,PublicInputGate_ins, 2);

            BaseSumGate<2> BaseSumGate_ins{ .num_limbs = 32 };
            DECL_GATE_NAME(BaseSumGate<2>,BaseSumGate_ins, 3);
            BaseSumGate<2> BaseSumGate_ins2{ .num_limbs = 63 };
            DECL_GATE_NAME(BaseSumGate<2>,BaseSumGate_ins2, 4);
            ArithmeticGate ArithmeticGate_ins{ .num_ops = 20 };
            DECL_GATE_NAME(ArithmeticGate,ArithmeticGate_ins, 5);
            BaseSumGate<4> BaseSumGate_ins3{ .num_limbs = 16 };
            DECL_GATE_NAME(BaseSumGate<4>,BaseSumGate_ins3, 6);

            ComparisonGate ComparisonGate_ins{ .num_bits = 32, .num_chunks = 16};
            DECL_GATE_NAME(ComparisonGate,ComparisonGate_ins, 7);

            U32AddManyGate U32AddManyGate_ins{ .num_addends = 0, .num_ops = 11};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins, 8);
            U32AddManyGate U32AddManyGate_ins2{ .num_addends = 11, .num_ops = 5};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins2, 9);
            U32AddManyGate U32AddManyGate_ins3{ .num_addends = 13, .num_ops = 5};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins3, 10);
            U32AddManyGate U32AddManyGate_ins4{ .num_addends = 15, .num_ops = 4};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins4, 11);
            U32AddManyGate U32AddManyGate_ins5{ .num_addends = 16, .num_ops = 4};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins5, 12);
            U32AddManyGate U32AddManyGate_ins6{ .num_addends = 2, .num_ops = 10};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins6, 13);
            U32AddManyGate U32AddManyGate_ins7{ .num_addends = 3, .num_ops = 9};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins7, 14);
            U32AddManyGate U32AddManyGate_ins8{ .num_addends = 5, .num_ops = 9};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins8, 15);
            U32AddManyGate U32AddManyGate_ins9{ .num_addends = 7, .num_ops = 8};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins9, 16);
            U32AddManyGate U32AddManyGate_ins10{ .num_addends = 9, .num_ops = 6};
            DECL_GATE_NAME(U32AddManyGate,U32AddManyGate_ins10, 17);
            U32ArithmeticGate U32ArithmeticGate_ins{ .num_ops = 6};
            DECL_GATE_NAME(U32ArithmeticGate,U32ArithmeticGate_ins, 18);
            U32RangeCheckGate U32RangeCheckGate_ins2{ .num_input_limbs = 0};
            DECL_GATE_NAME(U32RangeCheckGate,U32RangeCheckGate_ins2, 19);
            U32RangeCheckGate U32RangeCheckGate_ins3{ .num_input_limbs = 1};
            DECL_GATE_NAME(U32RangeCheckGate,U32RangeCheckGate_ins3, 20);
            U32RangeCheckGate U32RangeCheckGate_ins4{ .num_input_limbs = 8};
            DECL_GATE_NAME(U32RangeCheckGate,U32RangeCheckGate_ins4, 21);
            U32SubtractionGate U32SubtractionGate_ins{ .num_ops = 11};
            DECL_GATE_NAME(U32SubtractionGate,U32SubtractionGate_ins, 22);
            RandomAccessGate RandomAccessGate_ins{ .bits = 4, .num_copies = 4, .num_extra_constants = 2};
            DECL_GATE_NAME(RandomAccessGate,RandomAccessGate_ins, 23);
            PoseidonGate PoseidonGate_ins;
            DECL_GATE_NAME(PoseidonGate,PoseidonGate_ins, 24);

//            if (index == 1048576) {
//                printf("i: %d, local_constants: ", index);
//                local_constants.print_hex();
//                printf("i: %d, local_wires: ", index);
//                local_wires.print_hex();
//            }

            GoldilocksField terms[num_gate_constraints];
            auto evaluate_gate_constraints_base_batch = [index, public_inputs_hash, &constraint_terms_batch, &terms, gate_objs, selectors_info, local_constants, local_wires]() {
                for (int row = 0; row < num_gates; ++row) {
                    int selector_index = selectors_info.selector_indices[row];
                    auto gate = gate_objs[row];

                    auto compute_filter = [](int row, Range<int> group_range, GoldilocksField s,
                                             bool many_selector) -> GoldilocksField {
                        assert(group_range.contains(row));
                        GoldilocksField res = {1};
                        for (int i = group_range.first; i < group_range.second; ++i) {
                            if (i == row)
                                continue;
                            res *= GoldilocksField::from_canonical_u64(i) - s;
                        }

                        const uint32_t UNUSED_SELECTOR = UINT32_MAX;

                        if (many_selector) {
                            res *= GoldilocksField::from_canonical_u64(UNUSED_SELECTOR) - s;
                        }
                        return res;
                    };

                    auto filter = compute_filter(
                            row,
                            selectors_info.groups[selector_index],
                            local_constants[selector_index],
                            num_selectors > 1
                    );

                    EvaluationVarsBasePacked vars = {
                            .local_constants = local_constants.view(num_selectors, local_constants.len),
                            .local_wires = local_wires,
                            .public_inputs_hash = public_inputs_hash,
                            .index = index
                    };

//                    if (index == 1048576) {
//                        printf("i: %d, row: %d, filter: ", index, row);
//                        filter.print_hex(nullptr, GoldilocksField::newline);
//                    }

                    for (int i = 0; i < gate.num_constraints; ++i) {
                        terms[i] = GoldilocksField{0};
                    }

                    auto fn = gate.func;
                    auto yield_constr =  StridedConstraintConsumer{terms, &terms[gate.num_constraints]};
                    ((gate.gate)->*fn)(vars, yield_constr);
//                    if (index == 1048576) {
//                        printf("i: %d, row: %d, terms: ", index, row);
//                        GoldilocksFieldView{terms, gate.num_constraints}.print_hex();
//                    }

                    for (int i = 0; i < gate.num_constraints; ++i) {
                        constraint_terms_batch[i] += terms[i] * filter;
                    }

                }
            };

            evaluate_gate_constraints_base_batch();
        };
        evaluate_gate_constraints_base_batch();
//        if (index == 1048576) {
//            printf("i: %d, constraint_terms: ", index);
//            GoldilocksFieldView{constraint_terms_batch, num_gate_constraints}.print_hex();
//        }
        for (int i = num_gate_constraints-1; i >= 0; --i) {
            reduce_with_powers(constraint_terms_batch[i]);
        }

        constexpr int vanishing_partial_products_terms_len = num_challenges * num_routed_wires/max_degree;
        GoldilocksField vanishing_partial_products_terms[vanishing_partial_products_terms_len];
        for (int i = 0; i < num_challenges; ++i) {
            auto z_x = local_zs[i];
            auto z_gx = next_zs[i];

            // The partial products considered for this iteration of `i`.
            auto current_partial_products = partial_products.view(i * num_prods, (i + 1) * num_prods);
            // Check the numerator partial products.
//            let partial_product_checks = check_partial_products(
//                    &numerator_values,
//                    &denominator_values,
//                    current_partial_products,
////                    z_x,
////                    z_gx,
//                    max_degree,
//            );

            GoldilocksField prev_acc, next_acc;
            constexpr int partial_product_rounds = num_routed_wires/max_degree;
            assert(current_partial_products.len == partial_product_rounds-1);
            for (int k = 0; k < partial_product_rounds; ++k) {
                GoldilocksField num_chunk_product = GoldilocksField::from_canonical_u64(1);
                for (int j = k*max_degree; j < (k+1)*max_degree; ++j) {
                    auto wire_value = local_wires[j];
                    auto k_i = k_is[j];
                    auto s_id = k_i * x;
                    auto v = wire_value + betas[i] * s_id + gammas[i];
                    num_chunk_product *= v;
//                    if (index == 1048576) {
//                        printf("i: %d, wi: %d, ", index, j);
//                        wire_value.print_hex("wire_value", GoldilocksField::colum_space);
//                        k_i.print_hex("k_i", GoldilocksField::colum_space);
//                        x.print_hex("x", GoldilocksField::newline);
//                        v.print_hex("v", GoldilocksField::newline);
//                    }
                }
                GoldilocksField den_chunk_product = GoldilocksField::from_canonical_u64(1);
                for (int j = k*max_degree; j < (k+1)*max_degree; ++j) {
                    auto wire_value = local_wires[j];
                    auto s_sigma = s_sigmas[j];
                    den_chunk_product *= wire_value + betas[i] * s_sigma + gammas[i];
                }
                if (k == 0) {
                    prev_acc = z_x;
                } else {
                    prev_acc = next_acc;
                }

                if (k == partial_product_rounds-1)
                    next_acc = z_gx;
                else
                    next_acc = current_partial_products[k];

                vanishing_partial_products_terms[i * partial_product_rounds + k] = (prev_acc * num_chunk_product - next_acc * den_chunk_product);

//                if (index == 1048576) {
//                    printf("i: %d, partial_product_checks: ", index);
//                    GoldilocksFieldView{vanishing_partial_products_terms, vanishing_partial_products_terms_len}.print_hex();
//                }

            }
        }
//        if (index == 1048576) {
//            printf("i: %d, term: ", index);
//            GoldilocksFieldView{vanishing_partial_products_terms, vanishing_partial_products_terms_len}.print_hex();
//        }
        for (int i = vanishing_partial_products_terms_len-1; i >= 0; --i) {
            reduce_with_powers(vanishing_partial_products_terms[i]);
        }

        auto eval_l_0 = [z_h_on_coset_evals, rate_bits, degree_log](int index, GoldilocksField x) -> GoldilocksField {
//            if ((GoldilocksField::from_canonical_u64(1 << degree_log) * (x - GoldilocksField{1})).data == 0xfff923c55a2e4a87)
//                printf("index: %d\n", index);

            return z_h_on_coset_evals[index % (1<<rate_bits)] *
//                    (GoldilocksField::from_canonical_u64(1 << degree_log) * (x - GoldilocksField{1}));
                    (GoldilocksField::from_canonical_u64(1 << degree_log) * (x - GoldilocksField{1})).inverse();

        };

        auto l_0_x = eval_l_0(index, x);
//        if (index == 1048576) {
//            l_0_x.print_hex("l_0_x", GoldilocksField::colum_space);
//            z_h_on_coset_evals[index%(1<<rate_bits)].print_hex("ev", GoldilocksField::colum_space);
//            auto den = (GoldilocksField::from_canonical_u64(1<<degree_log) * (x - GoldilocksField{1}));
//            den.print_hex("den", GoldilocksField::colum_space);
//            den.inverse().print_hex("denv", GoldilocksField::newline);
//        }

        for (int i = num_challenges-1; i >= 0; --i) {
            auto z_x = local_zs[i];
//            res[0] = GoldilocksField::from_canonical_u64(0);
            reduce_with_powers(l_0_x * z_x.sub_one());
        }


        auto denominator_inv = z_h_on_coset_inverses[index % (1<<rate_bits)];
        for (int i = 0; i < num_challenges; ++i) {
            res[i] *= denominator_inv;
        }
//
//        if (index == 1048576) {
//            printf("i: %d, res: ", index);
//            GoldilocksFieldView{res, num_challenges}.print_hex();
//        }

        outs[index*2]   = res[0];
        outs[index*2+1] = res[1];
    }

}

__global__
void mul_kernel(GoldilocksField* values_flatten, int poly_num, int values_num_per_poly, const GoldilocksField* mul_values)
{
    int thCnt = get_global_thcnt();
    int gid = get_global_id();

    for (int i = gid; i < poly_num*values_num_per_poly; i += thCnt) {
        unsigned idx = i % values_num_per_poly;
        unsigned poly_idx = i / values_num_per_poly;

        GoldilocksField* values = values_flatten + poly_idx*values_num_per_poly;
//        if (idx == 2086137) {
//            printf("i: %d, poly:%d, res: ", idx, poly_idx);
//            values[idx].print_hex(nullptr, GoldilocksField::newline);
//        }

        values[idx] *= mul_values[idx];
    }
}

#endif
