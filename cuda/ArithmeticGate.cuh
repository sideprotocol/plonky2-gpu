
struct ArithmeticGate INHERIT_BASE {
    int num_ops;

    __device__ inline
    usize wire_ith_multiplicand_0(usize i) {
        return 4 * i;
    }
    __device__ inline
    usize wire_ith_multiplicand_1(usize i) {
        return 4 * i + 1;
    }
    __device__ inline
    usize wire_ith_addend(usize i) {
        return 4 * i + 2;
    }
    __device__ inline
    usize wire_ith_output(usize i) {
        return 4 * i + 3;
    }

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return num_ops;
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        auto const_0 = vars.local_constants[0];
        auto const_1 = vars.local_constants[1];

        for (int i = 0; i < num_ops; ++i) {
            auto multiplicand_0 = vars.local_wires[wire_ith_multiplicand_0(i)];
            auto multiplicand_1 = vars.local_wires[wire_ith_multiplicand_1(i)];
            auto addend = vars.local_wires[wire_ith_addend(i)];
            auto output = vars.local_wires[wire_ith_output(i)];
            auto computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;

            yield_constr.one(output - computed_output);
        }
    }


};

