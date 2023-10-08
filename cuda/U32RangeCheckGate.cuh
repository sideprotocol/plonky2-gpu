struct U32RangeCheckGate INHERIT_BASE {
    usize num_input_limbs;
    typedef U32RangeCheckGate Self;

    static constexpr usize AUX_LIMB_BITS = 2;
    static constexpr usize BASE = 1 << AUX_LIMB_BITS;

    __device__ inline
    usize aux_limbs_per_input_limb() const {
        return ceil_div_usize(32, AUX_LIMB_BITS); 
    }
    __device__ inline
    usize wire_ith_input_limb(usize i)  {
        assert(i < this->num_input_limbs);
        return i; 
    }
    __device__ inline
    usize wire_ith_input_limb_jth_aux_limb(usize i, usize j)  {
        assert(i < this->num_input_limbs);
        assert(j < this->aux_limbs_per_input_limb());
        return this->num_input_limbs + this->aux_limbs_per_input_limb() * i + j; 
    }

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return this->num_input_limbs * (1 + this->aux_limbs_per_input_limb());
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_one(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        auto base = GoldilocksField::from_canonical_usize(BASE);
        for (int i = 0; i < this->num_input_limbs; ++i) {
            auto input_limb = vars.local_wires[this->wire_ith_input_limb(i)];
            auto aux_limbs_range = Range<int>{0, this->aux_limbs_per_input_limb()};
            auto aux_limbs = [vars, this, i](int j) -> GoldilocksField {
                return vars.local_wires[this->wire_ith_input_limb_jth_aux_limb(i, j)];
            };
            auto computed_sum = reduce_with_powers(aux_limbs_range, aux_limbs, base);

            yield_constr.one(computed_sum - input_limb);
            for (auto j: aux_limbs_range) {
                auto aux_limb = aux_limbs(j);
                GoldilocksField product = {1};
                for (int k = 0; k < BASE; ++k) {
                    product *= aux_limb - GoldilocksField::from_canonical_usize(k);
                }
                yield_constr.one(product);
            }
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

