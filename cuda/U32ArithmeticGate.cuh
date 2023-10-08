struct U32ArithmeticGate INHERIT_BASE {
    usize num_ops;
    typedef U32ArithmeticGate Self;

    __device__ inline
    usize wire_ith_multiplicand_0(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i;
    }
    __device__ inline
    usize wire_ith_multiplicand_1(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i + 1;
    }
    __device__ inline
    usize wire_ith_addend(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i + 2;
    }

    __device__ inline
    usize wire_ith_output_low_half(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i + 3;
    }

    __device__ inline
    usize wire_ith_output_high_half(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i + 4;
    }

    __device__ inline
    usize wire_ith_inverse(usize i) {
        assert(i < this->num_ops);
        return routed_wires_per_op() * i + 5;
    }

    __device__ inline
    static constexpr usize limb_bits() {
        return 2;
    }
    __device__ inline
    static constexpr usize num_limbs() {
        return 64 / Self::limb_bits();
    }
    __device__ inline
    static constexpr usize routed_wires_per_op() {
        return 6;
    }
    __device__ inline
    usize wire_ith_output_jth_limb(usize i, usize j) {
        assert(i < this->num_ops);
        assert(j < Self::num_limbs());
        return routed_wires_per_op() * this->num_ops + Self::num_limbs() * i + j;
    }






    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return this->num_ops * (4 + Self::num_limbs());
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int i = 0; i < this->num_ops; ++i) {
            auto multiplicand_0 = vars.local_wires[this->wire_ith_multiplicand_0(i)];
            auto multiplicand_1 = vars.local_wires[this->wire_ith_multiplicand_1(i)];
            auto addend = vars.local_wires[this->wire_ith_addend(i)];

            auto computed_output = multiplicand_0 * multiplicand_1 + addend;

            auto output_low = vars.local_wires[this->wire_ith_output_low_half(i)];
            auto output_high = vars.local_wires[this->wire_ith_output_high_half(i)];
            auto inverse = vars.local_wires[this->wire_ith_inverse(i)];

            GoldilocksField combined_output;
            {
                auto base = GoldilocksField::from_canonical_u64(1ULL << 32);
                auto one = GoldilocksField{1};
                auto u32_max = GoldilocksField::from_canonical_u64(UINT32_MAX);

                // This is zero if and only if the high limb is `u32::MAX`.
                // u32::MAX - output_high
                auto diff = u32_max - output_high;
                // If this is zero, the diff is invertible, so the high limb is not `u32::MAX`.
                // inverse * diff - 1
                auto hi_not_max = inverse * diff - one;
                // If this is zero, either the high limb is not `u32::MAX`, or the low limb is zero.
                // hi_not_max * limb_0_u32
                auto hi_not_max_or_lo_zero = hi_not_max * output_low;

                yield_constr.one(hi_not_max_or_lo_zero);

                combined_output = output_high * base + output_low;
            }

            yield_constr.one(combined_output - computed_output);

            auto combined_low_limbs = GoldilocksField{0};
            auto combined_high_limbs = GoldilocksField{0};
            auto midpoint = Self::num_limbs() / 2;
            auto base = GoldilocksField::from_canonical_u64(1ULL << Self::limb_bits());
            for (int j = Self::num_limbs()-1; j >=0; --j) {
                auto this_limb = vars.local_wires[this->wire_ith_output_jth_limb(i, j)];
                auto max_limb = 1 << Self::limb_bits();
                GoldilocksField product = {1};
                for (int x = 0; x < max_limb; ++x) {
                    product *= this_limb - GoldilocksField::from_canonical_usize(x);
                }

                yield_constr.one(product);

                if (j < midpoint) {
                    combined_low_limbs = combined_low_limbs * base + this_limb;
                } else {
                    combined_high_limbs = combined_high_limbs * base + this_limb;
                }
            }
            yield_constr.one(combined_low_limbs - output_low);
            yield_constr.one(combined_high_limbs - output_high);
        }
    }
};

