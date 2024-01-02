struct NoopGate  INHERIT_BASE {
    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return 0;
    }


    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
    }

};

