/// Returns the index of the `i`th limb wire.


struct BaseSumGate : public Gate{
    usize num_limbs = 0;


    static constexpr usize WIRE_SUM = 0;
    static constexpr usize START_LIMBS = 1;

    __device__ inline
    Range<int> limbs() {
        return Range<int>{START_LIMBS, START_LIMBS + num_limbs};
    }

    __device__ inline
    virtual void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) override {
         auto sum = vars.local_wires[WIRE_SUM];
         auto limbs = vars.local_wires.view(this->limbs());
//         auto computed_sum = reduce_with_powers(limbs, F::from_canonical_usize(B));
//
//        yield_constr.one(computed_sum - sum);

//         constraints_iter = limbs.iter().map(|&limb| {
//                (0..B)
//                        .map(|i| limb - F::from_canonical_usize(i))
//                .product::<P>()
//        });
//        yield_constr.many(constraints_iter);

    }
};


