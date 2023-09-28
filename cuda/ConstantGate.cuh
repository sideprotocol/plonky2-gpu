struct ConstantGate INHERIT_BASE {
    usize num_consts;

    __device__ inline
    usize const_input(usize i) {
        assert(i < this->num_consts);
        return i;
    }

    __device__ inline
    usize wire_output(usize i) {
        assert(i < this->num_consts);
        return i;
    }

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return this->num_consts;
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int i = 0; i < num_consts; ++i) {
            yield_constr.one(vars.local_constants[this->const_input(i)] - vars.local_wires[this->wire_output(i)]);
        }
    }

};

