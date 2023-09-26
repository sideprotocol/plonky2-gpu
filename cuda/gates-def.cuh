#ifndef GATES_DEF_CUH
#define GATES_DEF_CUH

#include "def.cuh"

struct Gate {
    __device__ inline
    virtual int num_constraints() const = 0;
    __device__ inline
    virtual void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) = 0;

    __device__ inline
    virtual void eval_unfiltered_base_batch(
            EvaluationVarsBasePacked vars,
            GoldilocksField* constraints_batch,
            GoldilocksField* terms
    ) {

        eval_unfiltered_base_packed(
                vars,
                StridedConstraintConsumer{terms}
        );
    }
};

#include "ArithmeticGate.cuh"
#include "BaseSumGate.cuh"

__constant__ __device__ Gate* gates_g[] = {
        &ArithmeticGate_d
};


#endif

