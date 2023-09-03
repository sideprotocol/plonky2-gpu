#include <stdint.h>
#include <cassert>
#include <stdio.h>
#include <iostream>

//#define PRINT_HEX(data) \
//    do  {               \
//        printf("{");                \
//        for (int k = 0; k < sizeof(data); ++k) printf("0x%02x%s", ((uint8_t*)&(data))[k], k==sizeof(data)-1?"":", ");\
//        printf("}\n");\
//    } while(0)

#define PRINT_HEX_2(PROMT, ARR, N, BUF)					\
  do {									\
    int __my_local_remain = N;						\
    __my_local_remain -= snprintf(BUF, __my_local_remain, "%s: ", PROMT); \
    for (size_t __i_idx = 0; __i_idx < sizeof(ARR); __i_idx++) {	\
      __my_local_remain -= snprintf(BUF, __my_local_remain, "%02x", ((uint8_t*)&(ARR))[__i_idx]); \
      if ((__i_idx + 1) % 8 == 0 && __i_idx != sizeof(ARR) - 1) {	\
	__my_local_remain -= snprintf(BUF, __my_local_remain, ", ");	\
      }									\
    }									\
    snprintf(BUF, n, "\n");						\
  }while(0)

#define PRINT_HEX(PROMT, ARR)						\
  do {									\
    printf("%s: ", PROMT);						\
    for (size_t __i_idx = 0; __i_idx < sizeof(ARR); __i_idx++) {	\
      printf("%02x", ((uint8_t*)&(ARR))[__i_idx]);			\
      if ((__i_idx + 1) % 8 == 0 && __i_idx != sizeof(ARR) - 1) {	\
	printf(", ");							\
      }									\
    }									\
    printf("\n");							\
  }while(0)

static inline __device__ int get_global_id() {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    return gid;
}
static inline __device__ int get_global_thcnt()
{
    return gridDim.x * blockDim.x;
}

static inline __device__ uint64_t overflowing_add(uint64_t a, uint64_t b, int* overflow) {
    *overflow = UINT64_MAX - b < a;
    return a + b;
}
static inline __device__ uint64_t overflowing_sub(uint64_t a, uint64_t b, int* overflow) {
    *overflow = a < b;
    return a - b;
}

const uint64_t EPSILON = (1ULL << 32) - 1;

template<int BYTES>
struct __align__(8) bytes_pad_type {
    uint8_t data[BYTES];
};

#define BYTES_ASSIGN(dst, src, len)  \
        *(bytes_pad_type<len>*)(dst) = *(bytes_pad_type<len>*)(src)

typedef unsigned __int128 u128;

#if 0
class u128 {
public:
    uint64_t low;
    uint64_t high;

    __device__ inline u128(uint64_t l = 0, uint64_t h = 0) : low(l), high(h) {}

    __device__ inline u128 operator+(const u128& other) const {
        uint64_t sum_low = low + other.low;
        uint64_t carry = sum_low < low ? 1 : 0;
        uint64_t sum_high = high + other.high + carry;
        return u128(sum_low, sum_high);
    }

    __device__ inline u128 operator-(const u128& other) const {
        uint64_t diff_low = low - other.low;
        uint64_t borrow = diff_low > low ? 1 : 0;
        uint64_t diff_high = high - other.high - borrow;
        return u128(diff_low, diff_high);
    }

    __device__ inline u128 operator*(const u128& other) const {
//        uint64_t a0 = low & 0xFFFFFFFF;
//        uint64_t a1 = low >> 32;
//        uint64_t b0 = other.low & 0xFFFFFFFF;
//        uint64_t b1 = other.low >> 32;
//
//        uint64_t prod0 = a0 * b0;
//        uint64_t prod1 = a1 * b0 + (prod0 >> 32);
//        uint64_t prod2 = a0 * b1 + (prod1 & 0xFFFFFFFF);
//        uint64_t prod3 = a1 * b1 + (prod2 >> 32);
//
//        uint64_t carry = (prod3 >> 32) + (prod2 >> 32) + (prod1 >> 32);
//        uint64_t result_low = (prod0 & 0xFFFFFFFF) | (prod1 << 32);
//        uint64_t result_high = prod3 + carry;
//
//        return u128{result_low, result_high};
        auto b = other;
        auto a = *this;

        u128 result = {0, 0};
        for (int i = 0; i < 64; i++) {
            if (b.low & 1) {
                result = result + a;
            }
            a = a << 1;
            b.low >>= 1;
        }
        return result;

    }

    __device__ inline u128& operator+=(const u128& other) {
        *this = *this + other;
        return *this;
    }

    __device__ inline u128 operator>>(int shift) const {
        if (shift >= 128) {
            return u128(0, 0);
        } else if (shift >= 64) {
            return u128(high >> (shift - 64), 0);
        } else {
            return u128((low >> shift) | (high << (64 - shift)), high >> shift);
        }
    }

    __device__ inline u128 operator<<(int shift) const {
        u128 result;
        if (shift >= 64) {
            result.high = this->low << (shift - 64);
            result.low = 0;
        } else {
            result.high = (this->high << shift) | (this->low >> (64 - shift));
            result.low = this->low << shift;
        }
        return result;
        return result;
    }

    __device__ inline u128 overflowing_add(const u128& other, bool* overflow) const {
        u128 result = *this + other;
        *overflow = (result.high < high) || ((result.high == high) && (result.low < low));
        return result;
    }

    __device__ inline operator uint64_t() const {
        return low;
    }

};
#endif

struct  GoldilocksField{
    uint64_t data;
    static const uint64_t ORDER = 0xFFFFFFFF00000001;

    __device__ inline GoldilocksField square() const {
        return (*this) * (*this);
    }
    __device__ inline uint64_t to_noncanonical_u64() const{
        return this->data;
    }

    static __device__ inline GoldilocksField from_canonical_u64(uint64_t n) {
        return GoldilocksField{n};
    }

    static __device__ inline GoldilocksField from_noncanonical_u96(uint64_t n_lo, uint32_t n_hi) {
        // Default implementation.
        u128 n = (u128(n_hi) << 64) + u128(n_lo);
        return from_noncanonical_u128(n);
    }

    static __device__ inline GoldilocksField  from_noncanonical_u128(u128 n) {
        return reduce128(n >> 64, n & UINT64_MAX);
    }

    __device__ inline
    GoldilocksField operator+(const GoldilocksField& rhs) {
        int over = 0;
        uint64_t sum = overflowing_add(this->data, rhs.data, &over);
        sum = overflowing_add(sum, over * EPSILON, &over);
        if (over) {
                // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
                // double-overflow.
                // This assume does two things:
                //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this
                //     check.
                //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
                //     a branch).
                assert(this->data > GoldilocksField::ORDER && rhs.data > GoldilocksField::ORDER);
//                    branch_hint();
                sum += EPSILON; // Cannot overflow.
        }
        return GoldilocksField{.data = sum};
    }
    __device__ inline
    GoldilocksField operator-(const GoldilocksField& rhs) {
        int under = 0;
        uint64_t diff = overflowing_sub(this->data, rhs.data, &under);
        diff = overflowing_sub(diff, under * EPSILON, &under);
        if (under) {
            // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this
            //     check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            assert(this->data < EPSILON - 1 && rhs.data > GoldilocksField::ORDER);
//                    branch_hint();
            diff -= EPSILON; // Cannot overflow.
        }
        return GoldilocksField{.data = diff};
    }

    static __device__ inline
    GoldilocksField reduce128(uint64_t x_hi, uint64_t x_lo) {
        uint64_t x_hi_hi = x_hi >> 32;
        uint64_t x_hi_lo = x_hi & EPSILON;

        int borrow = 0;
        uint64_t t0 = overflowing_sub(x_lo, x_hi_hi, &borrow);
        if (borrow) {
//            branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
            t0 -= EPSILON; // Cannot underflow.
        }
        uint64_t t1 = x_hi_lo * EPSILON;
//        uint64_t t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };

        uint64_t t2;
        if (UINT64_MAX - t1 < t0) {
            t2 = (t1 + t0) + (0xffffffff);
        }
        else
            t2 = (t0 + t1);
        return GoldilocksField{.data = t2};
    }

    __device__ inline
    GoldilocksField operator*(const GoldilocksField& rhs) const {
        uint64_t high, low, a = this->data, b = rhs.data;
        {
            uint64_t a_low = a & 0xFFFFFFFF;
            uint64_t a_high = a >> 32;
            uint64_t b_low = b & 0xFFFFFFFF;
            uint64_t b_high = b >> 32;

            uint64_t product_low = a_low * b_low;
            uint64_t product_mid1 = a_low * b_high;
            uint64_t product_mid2 = a_high * b_low;
            uint64_t product_high = a_high * b_high;

            uint64_t carry = (product_low >> 32) + (product_mid1 & 0xFFFFFFFF) + (product_mid2 & 0xFFFFFFFF);
            high = product_high + (product_mid1 >> 32) + (product_mid2 >> 32) + (carry >> 32);
            low = (carry << 32) + (product_low & 0xFFFFFFFF);
        }
        return reduce128(high, low);
    }
    __device__ inline
    GoldilocksField& operator*=(const GoldilocksField& rhs) {
        *this = *this * rhs;
        return *this;
    }
    __device__ inline
    GoldilocksField& operator+=(const GoldilocksField& rhs) {
        *this = *this + rhs;
        return *this;
    }

    __device__ inline
    GoldilocksField multiply_accumulate(GoldilocksField x, GoldilocksField y) {
        // Default implementation.
        return *this + x * y;
    }

    __device__ inline
    GoldilocksField add_canonical_u64(uint64_t rhs) {
        // Default implementation.
        return *this + GoldilocksField::from_canonical_u64(rhs);
    }

};

#include "constants.cuh"



struct PoseidonHasher {
    typedef uint32_t u32;
    typedef uint64_t u64;

    struct HashOut {
        GoldilocksField elements[4] ;
    };

    template<class T1, class T2>
    struct my_pair {
        T1 first;
        T2 second;
        __device__ inline my_pair(const T1& t1, const T2& t2)
        :first(t1), second(t2)
        {
        }
    };

    static __device__ inline my_pair<u128, u32> add_u160_u128(my_pair<u128, u32> pa, u128 y) {
        auto x_lo = pa.first;
        auto x_hi = pa.second;

         auto overflowing_add = [](u128 a, u128 b, bool* overflow) {
            *overflow = ~__uint128_t{} - b < a;
            return a + b;
        };

        bool over;
        auto res_lo = overflowing_add(x_lo, y, &over);
        u32 res_hi = x_hi + u32(over);
        return my_pair<u128, u32>{res_lo, res_hi};
    }

    static __device__ inline GoldilocksField reduce_u160(my_pair<u128, u32> pa) {
        auto n_lo = pa.first;
        auto n_hi = pa.second;

        u64 n_lo_hi = (n_lo >> 64);
        u64 n_lo_lo = n_lo;
        u64 reduced_hi = GoldilocksField::from_noncanonical_u96(n_lo_hi, n_hi).to_noncanonical_u64();
        u128 reduced128 = (u128(reduced_hi) << 64) + u128(n_lo_lo);
        return GoldilocksField::from_noncanonical_u128(reduced128);
    }

    static __device__ inline void print_state(const char* promt, GoldilocksField* state) {
        printf("%s: [", promt);
        for (int i = 0; i < 12; ++i) {
            printf("%lu%s", state[i], i == 11?"]\n":", ");
        }
    }
    static __device__ inline
    void permute_poseidon(GoldilocksField* state) {
        int round_ctr = 0;

        constexpr int WIDTH = SPONGE_WIDTH;
        auto constant_layer = [&]() {
            for (int i = 0; i < 12; ++i) {
                if (i < WIDTH) {
                    uint64_t round_constant = ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr];
                    state[i] = state[i].add_canonical_u64(round_constant);
                }
            }
        };

        auto sbox_monomial = [](GoldilocksField x) -> GoldilocksField {
            // x |--> x^7
            GoldilocksField x2 = x.square();
            GoldilocksField x4 = x2.square();
            GoldilocksField x3 = x * x2;
            return x3 * x4;
        };

        auto sbox_layer = [&]() {
            for (int i = 0; i < 12; ++i) {
                if (i < WIDTH) {
                    state[i] = sbox_monomial(state[i]);
                }
            }
        };

        auto mds_row_shf = [](int r, uint64_t v[WIDTH]) -> u128 {
            assert(r < WIDTH);
            // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
            // known to be small, so we can accumulate all the products for
            // each row and reduce just once at the end (done by the
            // caller).

            // NB: Unrolling this, calculating each term independently, and
            // summing at the end, didn't improve performance for me.
            u128 res = 0;

            // This is a hacky way of fully unrolling the loop.
            for (int i = 0; i < 12; ++i) {
                if (i < WIDTH) {
                    res += u128(v[(i + r) % WIDTH]) * u128(MDS_MATRIX_CIRC[i]);
//                    printf("state 1211: %lu, %lu\n", res.high, res.low);
                }
            }
            res += u128(v[r]) * u128(MDS_MATRIX_DIAG[r]);
            return res;
        };

        auto mds_layer = [&]() {
            uint64_t _state[SPONGE_WIDTH] = {0};

            for (int r = 0; r < WIDTH; ++r)
                _state[r] = state[r].to_noncanonical_u64();

            // This is a hacky way of fully unrolling the loop.
            for (int r = 0; r < 12; ++r) {
                if (r < WIDTH) {
                    auto sum = mds_row_shf(r, _state);
//                    printf("state 121: %lu, %lu\n", sum.high, sum.low);
                    uint64_t sum_lo = sum;
                    uint32_t sum_hi = (sum >> 64);
                    state[r] = GoldilocksField::from_noncanonical_u96(sum_lo, sum_hi);
//                    printf("state 122: %lu, lo: %lu, hi: %u\n", state[r].data, sum_lo, sum_hi);
                }
            }
        };

        auto full_rounds = [&]() {
            for (int r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
                constant_layer();
//                print_state("state11", state);
                sbox_layer();
//                print_state("state12", state);
                mds_layer();
//                print_state("state13", state);
                round_ctr += 1;
            }
        };

        auto partial_first_constant_layer = [&]() {
            for (int i = 0; i < 12; ++i) {
                if (i < WIDTH) {
                    state[i] += GoldilocksField::from_canonical_u64(FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
                }
            }
        };

        auto mds_partial_layer_init = [&]() {
            // Initial matrix has first row/column = [1, 0, ..., 0];

            GoldilocksField result[WIDTH] = {0};
            // c = 0
            result[0] = state[0];

            for (int r = 1; r < 12; ++r) {
                if (r < WIDTH) {
                    for (int c = 1; c < 12; ++c) {
                        if (c < WIDTH) {
                            // NB: FAST_PARTIAL_ROUND_INITIAL_MATRIX is stored in
                            // row-major order so that this dot product is cache
                            // friendly.
                            auto t = GoldilocksField::from_canonical_u64(
                                    FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1]
                            );
                            result[c] += state[r] * t;
                        }
                    }
                }
            }
            for (int i = 0; i < WIDTH; ++i)
                state[i] = result[i];
        };

        auto mds_partial_layer_fast = [&](int r) {
            // Set d = [M_00 | w^] dot [state]
//            print_state("state21", state);

            my_pair<u128, u32> d_sum = {0, 0}; // u160 accumulator
            for (int i = 1; i < 12; ++i) {
                if (i < WIDTH) {
                    u128 t = FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
                    u128 si = state[i].to_noncanonical_u64();
                    d_sum = add_u160_u128(d_sum, si * t);
                }
            }

            u128 s0 = u128(state[0].to_noncanonical_u64());
            u128 mds0to0 = u128(MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]);
            d_sum = add_u160_u128(d_sum, s0 * mds0to0);
            auto d = reduce_u160(d_sum);

            // result = [d] concat [state[0] * v + state[shift up by 1]]
            GoldilocksField result[SPONGE_WIDTH];
//            let mut result = [ZERO; WIDTH];
            result[0] = d;
            for (int i = 1; i < 12; ++i) {
                if (i < WIDTH) {
                    auto t = GoldilocksField::from_canonical_u64(FAST_PARTIAL_ROUND_VS[r][i - 1]);
                    result[i] = state[i].multiply_accumulate(state[0], t);
                }
            }
            for (int i = 0; i < 12; ++i)
                state[i] = result[i];
//            print_state("state22", state);
        };

        auto partial_rounds = [&]() {
            partial_first_constant_layer();
            mds_partial_layer_init();

            for (int i = 0; i < N_PARTIAL_ROUNDS; ++i) {
                state[0] = sbox_monomial(state[0]);
//            unsafe
                {
                    state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
                }
//                *state = mds_partial_layer_fast(state, i);
                mds_partial_layer_fast(i);
            }
            round_ctr += N_PARTIAL_ROUNDS;
        };

//        print_state("state1", state);
        full_rounds();
//        print_state("state2", state);
        partial_rounds();
//        print_state("state3", state);
        full_rounds();
//        print_state("state4", state);

        assert(round_ctr == N_ROUNDS);

    }

    static __device__ inline HashOut hash_n_to_m_no_pad(const GoldilocksField* input) {
        GoldilocksField state[SPONGE_WIDTH] = {0};

        constexpr int len = 4;
        // Absorb all input chunks.
        for (int i = 0; i < len; i += SPONGE_RATE) {
            for (int j = 0; j < SPONGE_RATE; ++j)
                state[j] = input[i*SPONGE_RATE+j];
            permute_poseidon(state);
        }

        return *(HashOut*)state;
    }
};

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
        }
//        if (get_global_id() == 0) {
//            printf("buf5 lg_half_m:%d: ", lg_half_m);
//            for (int i = (1<<20); i < 8+(1<<20); ++i) {
//                printf("%016lX, ", packed_values[i].data);
//            }
//            printf("\n");
//        }
        __syncthreads();
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
    fft_dispatch(values_flatten, poly_num, values_num_per_poly, log_len, root_table, 0);

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

    assert(thCnt > poly_num);

    for (int i = gid; i < poly_num*values_num_per_poly; i += thCnt) {
        unsigned val_idx = i / poly_num;
        unsigned poly_idx = i % poly_num;

        GoldilocksField *src_value = src_values_flatten + poly_idx * values_num_per_poly + val_idx;
        GoldilocksField *dst_value = dst_values_flatten + val_idx * poly_num + poly_idx;

        *dst_value = *src_value;
    }
}

#endif
