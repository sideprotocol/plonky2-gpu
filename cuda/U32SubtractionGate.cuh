struct U32SubtractionGate INHERIT_BASE {
    usize num_ops;
    typedef U32SubtractionGate Self;



    __device__ inline
    usize wire_ith_input_x(usize i) {
        assert(i < this->num_ops);
        return 5 * i; 
    }
    __device__ inline
    usize wire_ith_input_y(usize i) {
        assert(i < this->num_ops);
        return 5 * i + 1; 
    }
    __device__ inline
    usize wire_ith_input_borrow(usize i) {
        assert(i < this->num_ops);
        return 5 * i + 2; 
    }

    __device__ inline
    usize wire_ith_output_result(usize i) {
        assert(i < this->num_ops);
        return 5 * i + 3; 
    }
    __device__ inline
    usize wire_ith_output_borrow(usize i) {
        assert(i < this->num_ops);
        return 5 * i + 4; 
    }

    __device__ inline
    static constexpr usize limb_bits() {
        return 2; 
    }
    // We have limbs for the 32 bits of `output_result`.
    __device__ inline
    static constexpr usize num_limbs() {
        return 32 / Self::limb_bits(); 
    }

    __device__ inline
    usize wire_ith_output_jth_limb(usize i, usize j) {
        assert(i < this->num_ops);
        assert(j < Self::num_limbs());
        return 5 * this->num_ops + Self::num_limbs() * i + j; 
    }





    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return this->num_ops * (3 + Self::num_limbs());
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int i = 0; i < this->num_ops; ++i) {
            auto input_x = vars.local_wires[this->wire_ith_input_x(i)];
            auto input_y = vars.local_wires[this->wire_ith_input_y(i)];
            auto input_borrow = vars.local_wires[this->wire_ith_input_borrow(i)];

            auto result_initial = input_x - input_y - input_borrow;
            auto base = GoldilocksField::from_canonical_u64(1ULL << 32);

            auto output_result = vars.local_wires[this->wire_ith_output_result(i)];
            auto output_borrow = vars.local_wires[this->wire_ith_output_borrow(i)];

            yield_constr.one(output_result - (result_initial + output_borrow * base));

            // Range-check output_result to be at most 32 bits.
            auto combined_limbs = GoldilocksField{0};
            auto limb_base = GoldilocksField::from_canonical_u64(1ULL << Self::limb_bits());
            for (int j = Self::num_limbs()-1; j >=0; --j) {
                auto this_limb = vars.local_wires[this->wire_ith_output_jth_limb(i, j)];
                auto max_limb = 1 << Self::limb_bits();
                GoldilocksField product = {1};
                for (int x = 0; x < max_limb; ++x) {
                    product *= this_limb - GoldilocksField::from_canonical_usize(x);
                }
                yield_constr.one(product);

                combined_limbs = combined_limbs * limb_base + this_limb;
            }
            yield_constr.one(combined_limbs - output_result);

            // Range-check output_borrow to be one bit.
            yield_constr.one(output_borrow * (GoldilocksField{1} - output_borrow));
        }
    }
};

