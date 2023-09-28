struct PublicInputGate :public Gate {

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return 4;
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int i = 0; i < 4; ++i) {
            auto wire = i;
            auto hash_part = vars.public_inputs_hash.elements[i];
            yield_constr.one(vars.local_wires[wire] - hash_part);
        }
    }

};

