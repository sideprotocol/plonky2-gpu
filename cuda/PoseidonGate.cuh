
struct Poseidon {
    struct HashOut {
        GoldilocksField elements[4] ;
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
            printf("%lu%s", state[i].data, i == 11?"]\n":", ");
        }
    }

    static constexpr int WIDTH = SPONGE_WIDTH;


    static __device__ inline
    void constant_layer(GoldilocksField* state, int &round_ctr) {
        for (int i = 0; i < 12; ++i) {
            if (i < WIDTH) {
                uint64_t round_constant = ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr];
                state[i] = state[i].add_canonical_u64(round_constant);
            }
        }
    }

    static __device__ inline
    GoldilocksField sbox_monomial(GoldilocksField x) {
        // x |--> x^7
        GoldilocksField x2 = x.square();
        GoldilocksField x4 = x2.square();
        GoldilocksField x3 = x * x2;
        return x3 * x4;
    }

    static __device__ inline
    void sbox_layer(GoldilocksField* state) {
        for (int i = 0; i < 12; ++i) {
            if (i < WIDTH) {
                state[i] = sbox_monomial(state[i]);
            }
        }
    }

    static __device__ inline
    void mds_layer(GoldilocksField* state) {
        uint64_t _state[SPONGE_WIDTH] = {0};
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
    }

    static __device__ inline
    void partial_first_constant_layer(GoldilocksField* state) {
        for (int i = 0; i < 12; ++i) {
            if (i < WIDTH) {
                state[i] += GoldilocksField::from_canonical_u64(FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
            }
        }
    }

    static __device__ inline
    void mds_partial_layer_init(GoldilocksField* state) {
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
    }

    static __device__ inline
    void mds_partial_layer_fast(GoldilocksField* state, int r) {
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
    }

};

struct PoseidonGate INHERIT_BASE {
    typedef PoseidonGate Self;


    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return SPONGE_WIDTH * (N_FULL_ROUNDS_TOTAL - 1)
           + N_PARTIAL_ROUNDS
           + SPONGE_WIDTH
           + 1
           + 4;
    }

    /// The wire index for the `i`th input to the permutation.
    __device__ inline
    usize wire_input(usize i) {
        return i; 
    }

    /// The wire index for the `i`th output to the permutation.
    __device__ inline
    usize wire_output(usize i) {
        return SPONGE_WIDTH + i; 
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    static constexpr usize WIRE_SWAP = 2 * SPONGE_WIDTH;

    static constexpr usize START_DELTA = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    __device__ inline
    usize wire_delta(usize i) {
        assert(i < 4);
        return Self::START_DELTA + i; 
    }

    static constexpr usize START_FULL_0 = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    __device__ inline
    usize wire_full_sbox_0(usize round, usize i) {
        assert(
                round != 0
//                        "First round S-box inputs are not stored as wires"
        );
        assert(round < HALF_N_FULL_ROUNDS);
        assert(i < SPONGE_WIDTH);
        return Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i; 
    }

    static constexpr  usize START_PARTIAL =
            Self::START_FULL_0 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    __device__ inline
    usize wire_partial_sbox(usize round) {
        assert(round < N_PARTIAL_ROUNDS);
        return Self::START_PARTIAL + round; 
    }

    static constexpr  usize START_FULL_1 = Self::START_PARTIAL + N_PARTIAL_ROUNDS;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    __device__ inline
    usize wire_full_sbox_1(usize round, usize i) {
        assert(round < HALF_N_FULL_ROUNDS);
        assert(i < SPONGE_WIDTH);
        return Self::START_FULL_1 + SPONGE_WIDTH * round + i; 
    }

    /// End of wire indices, exclusive.
    __device__ inline
    usize end() {
        return Self::START_FULL_1 + SPONGE_WIDTH * HALF_N_FULL_ROUNDS; 
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_one(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        // Assert that `swap` is binary.
        auto swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for (int i = 0; i < 4; ++i) {
            auto input_lhs = vars.local_wires[Self::wire_input(i)];
            auto input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            auto delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        GoldilocksField state[SPONGE_WIDTH] = {0};
        for (int i = 0; i < 4; ++i) {
            auto delta_i = vars.local_wires[Self::wire_delta(i)];
            auto input_lhs = Self::wire_input(i);
            auto input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for (int i = 8; i < SPONGE_WIDTH; ++i) {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        int round_ctr = 0;

        // First set of full rounds.
        for (int r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
                Poseidon::constant_layer(state, round_ctr);
                if (r != 0) {
                    for (int i = 0; i < SPONGE_WIDTH; ++i) {
                        auto sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                        yield_constr.one(state[i] - sbox_in);
                        state[i] = sbox_in;
                    }
                }
                Poseidon::sbox_layer(state);
                Poseidon::mds_layer(state);
                round_ctr += 1;
        }

                            // Partial rounds.
                            Poseidon::partial_first_constant_layer(state);
        Poseidon::mds_partial_layer_init(state);
        for (int r = 0; r < (N_PARTIAL_ROUNDS - 1); ++r) {
            auto sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = Poseidon::sbox_monomial(sbox_in);
            state[0] += GoldilocksField::from_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[r]);
            Poseidon::mds_partial_layer_fast(state, r);
        }
        auto sbox_in = vars.local_wires[Self::wire_partial_sbox(N_PARTIAL_ROUNDS - 1)];
        yield_constr.one(state[0] - sbox_in);
        state[0] = Poseidon::sbox_monomial(sbox_in);
        Poseidon::mds_partial_layer_fast(state, N_PARTIAL_ROUNDS - 1);
        round_ctr += N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for (int r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
                Poseidon::constant_layer(state, round_ctr);
                for (int i = 0; i < SPONGE_WIDTH; ++i) {
                    auto sbox_in2 = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                    yield_constr.one(state[i] - sbox_in2);
                    state[i] = sbox_in2;
                }
                Poseidon::sbox_layer(state);
                Poseidon::mds_layer(state);
                round_ctr += 1;
        }

        for (int i = 0; i < SPONGE_WIDTH; ++i) {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        eval_unfiltered_base_one(
                vars,
                yield_constr
        );
    }
};

