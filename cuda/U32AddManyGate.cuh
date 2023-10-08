
constexpr usize LOG2_MAX_NUM_ADDENDS = 4;
constexpr usize MAX_NUM_ADDENDS = 16;

struct U32AddManyGate INHERIT_BASE {
    usize num_addends;
    usize num_ops;
    typedef U32AddManyGate Self;

//    usize num_ops(num_addends: usize, config: &CircuitConfig) {
//        assert(num_addends <= MAX_NUM_ADDENDS);
//        let wires_per_op = (num_addends + 3) + Self::num_limbs();
//        let routed_wires_per_op = num_addends + 3;
//        (config.num_wires / wires_per_op).min(config.num_routed_wires / routed_wires_per_op)
//    }

    __device__ inline
    static constexpr usize limb_bits() {
      return 2; 
    }
    __device__ inline
    static constexpr usize num_result_limbs() {
      return ceil_div_usize(32, Self::limb_bits());
    }
    __device__ inline
    static constexpr usize num_carry_limbs() {
      return ceil_div_usize(LOG2_MAX_NUM_ADDENDS, Self::limb_bits()); 
    }
    __device__ inline
    static constexpr usize num_limbs() {
      return Self::num_result_limbs() + Self::num_carry_limbs(); 
    }

    __device__ inline
    usize wire_ith_op_jth_addend(usize i, usize j) {
        assert(i < this->num_ops);
        assert(j < this->num_addends);
        return (this->num_addends + 3) * i + j; 
    }
    __device__ inline
    usize wire_ith_carry(usize i) {
        assert(i < this->num_ops);
        return (this->num_addends + 3) * i + this->num_addends; 
    }

    __device__ inline
    usize wire_ith_output_result(usize i) {
        assert(i < this->num_ops);
        return (this->num_addends + 3) * i + this->num_addends + 1; 
    }
    __device__ inline
    usize wire_ith_output_carry(usize i) {
        assert(i < this->num_ops);
        return (this->num_addends + 3) * i + this->num_addends + 2; 
    }

    __device__ inline
    usize wire_ith_output_jth_limb(usize i, usize j) {
        assert(i < this->num_ops);
        assert(j < Self::num_limbs());
        return (this->num_addends + 3) * this->num_ops + Self::num_limbs() * i + j; 
    }


    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return this->num_ops * (3 + Self::num_limbs());
    }



    __device__ inline
    VIRTUAL void eval_unfiltered_base_one(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int i = 0; i < this->num_ops; ++i) {
            auto carry = vars.local_wires[this->wire_ith_carry(i)];

            GoldilocksField computed_output = {0};
            for (int j = 0; j< this->num_addends; ++j) {
                auto y = vars.local_wires[this->wire_ith_op_jth_addend(i, j)];
                computed_output += y;
            }
            computed_output += carry;

            auto output_result = vars.local_wires[this->wire_ith_output_result(i)];
            auto output_carry = vars.local_wires[this->wire_ith_output_carry(i)];

            auto base0 = GoldilocksField::from_canonical_u64(1ULL << 32);
            auto combined_output = output_carry * base0 + output_result;

            yield_constr.one(combined_output - computed_output);
            auto combined_result_limbs = GoldilocksField{0};
            auto combined_carry_limbs = GoldilocksField{0};
            auto base = GoldilocksField::from_canonical_u64(1ULL << Self::limb_bits());
            for (int j = Self::num_limbs()-1; j >=0; --j) {
                auto this_limb = vars.local_wires[this->wire_ith_output_jth_limb(i, j)];
                auto max_limb = 1 << Self::limb_bits();
                GoldilocksField product = {1};
                for (int x = 0; x < max_limb; ++x) {
                    product *= this_limb - GoldilocksField::from_canonical_usize(x);
                }
                yield_constr.one(product);

                if (j < Self::num_result_limbs()) {
                    combined_result_limbs = base * combined_result_limbs + this_limb;
                } else {
                    combined_carry_limbs = base * combined_carry_limbs + this_limb;
                }
            }
            yield_constr.one(combined_result_limbs - output_result);
            yield_constr.one(combined_carry_limbs - output_carry);
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


