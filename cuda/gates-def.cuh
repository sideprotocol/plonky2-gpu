#ifndef GATES_DEF_CUH
#define GATES_DEF_CUH

#include "def.cuh"

struct Gate {
    __device__ inline
    virtual int num_constraints() const = 0;
    __device__ inline
    virtual void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) {};

    __device__ inline
    virtual void eval_unfiltered_base_one(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) {};

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

#define USE_VIRTUAL_FUN 0

#if USE_VIRTUAL_FUN
#define INHERIT_BASE  : public Gate
#define VIRTUAL virtual
#define OVERRIDE override
#else
#define INHERIT_BASE
#define VIRTUAL
#define OVERRIDE

#endif

#include "ArithmeticGate.cuh"
#include "BaseSumGate.cuh"

#include "ComparisonGate.cuh"
#include "ConstantGate.cuh"
#include "NoopGate.cuh"
#include "PoseidonGate.cuh"
#include "PublicInputGate.cuh"
#include "RandomAccessGate.cuh"
#include "U32AddManyGate.cuh"
#include "U32ArithmeticGate.cuh"
#include "U32RangeCheckGate.cuh"
#include "U32SubtractionGate.cuh"



#endif

