/// Returns the index of the `i`th limb wire.


template<int B>
struct BaseSumGate  INHERIT_BASE{
    usize num_limbs;


    static constexpr usize WIRE_SUM = 0;
    static constexpr usize START_LIMBS = 1;

    __device__ inline
    Range<int> limbs() {
        return Range<int>{START_LIMBS, START_LIMBS + num_limbs};
    }

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return 1 + this->num_limbs;
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
         auto sum = vars.local_wires[WIRE_SUM];
         auto limbs = vars.local_wires.view(this->limbs());
         auto computed_sum = reduce_with_powers(limbs, GoldilocksField::from_canonical_u64(B));

        yield_constr.one(computed_sum - sum);

        for (auto limb: limbs) {
            GoldilocksField product = GoldilocksField{1};
            for (int i = 0; i < B; ++i) {
                product *= limb - GoldilocksField::from_canonical_u64(i);
            }
            yield_constr.one(product);
        }

    }
};


