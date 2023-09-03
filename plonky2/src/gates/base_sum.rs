use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::ops::Range;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::field::types::{Field, Field64};
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CircuitConfig;
use crate::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::log_floor;

/// A gate which can decompose a number into base B little-endian limbs.
#[derive(Copy, Clone, Debug)]
pub struct BaseSumGate<const B: usize> {
    pub num_limbs: usize,
}

impl<const B: usize> BaseSumGate<B> {
    pub fn new(num_limbs: usize) -> Self {
        Self { num_limbs }
    }

    pub fn new_from_config<F: Field64>(config: &CircuitConfig) -> Self {
        let num_limbs =
            log_floor(F::ORDER - 1, B as u64).min(config.num_routed_wires - Self::START_LIMBS);
        Self::new(num_limbs)
    }

    pub const WIRE_SUM: usize = 0;
    pub const START_LIMBS: usize = 1;

    /// Returns the index of the `i`th limb wire.
    pub fn limbs(&self) -> Range<usize> {
        Self::START_LIMBS..Self::START_LIMBS + self.num_limbs
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const B: usize> Gate<F, D> for BaseSumGate<B> {
    fn id(&self) -> String {
        format!("{self:?} + Base: {B}")
    }

    fn export_circom_verification_code(&self) -> String {
        let mut template_str = format!(
            "template BaseSum$NUM_LIMBS() {{
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  $SET_FILTER;

  component reduce = Reduce($NUM_LIMBS);
  reduce.alpha <== GlExt($B, 0)();
  reduce.old_eval <== GlExt(0, 0)();
  for (var i = 1; i < $NUM_LIMBS + 1; i++) {{
    reduce.in[i - 1] <== wires[i];
  }}
  out[0] <== ConstraintPush()(constraints[0], filter, GlExtSub()(reduce.out, wires[0]));
  component product[$NUM_LIMBS][$B - 1];
  for (var i = 0; i < $NUM_LIMBS; i++) {{
    for (var j = 0; j < $B - 1; j++) {{
      product[i][j] = GlExtMul();
      if (j == 0) product[i][j].a <== wires[i + 1];
      else product[i][j].a <== product[i][j - 1].out;
      product[i][j].b <== GlExtSub()(wires[i + 1], GlExt(j + 1, 0)());
    }}
    out[i + 1] <== ConstraintPush()(constraints[i + 1], filter, product[i][$B - 2].out);
  }}
  for (var i = $NUM_LIMBS + 1; i < NUM_GATE_CONSTRAINTS(); i++) {{
    out[i] <== constraints[i];
  }}
}}"
        )
        .to_string();
        template_str = template_str.replace("$NUM_LIMBS", &*self.num_limbs.to_string());
        template_str = template_str.replace("$B", &*B.to_string());

        template_str
    }
    fn export_solidity_verification_code(&self) -> String {
        let mut template_str = format!("library BaseSum$NUM_LIMBSLib {{
    using GoldilocksExtLib for uint64[2];
    function set_filter(GatesUtilsLib.EvaluationVars memory ev) internal pure {{
        $SET_FILTER;
    }}
    function eval(GatesUtilsLib.EvaluationVars memory ev, uint64[2][$NUM_GATE_CONSTRAINTS] memory constraints) internal pure {{
        uint64[2] memory sum;
        for (uint32 i = $NUM_LIMBS + 1; i > 1; i--) {{
            sum = sum.mul(GatesUtilsLib.field_ext_from($B, 0)).add(ev.wires[i - 1]);
        }}
        GatesUtilsLib.push(constraints, ev.filter, 0, sum.sub(ev.wires[0]));
        for (uint32 i = 1; i < $NUM_LIMBS + 1; i++) {{
            uint64[2] memory product = ev.wires[i];
            for (uint32 j = 1; j < $B; j++) {{
                product = product.mul(ev.wires[i].sub(GatesUtilsLib.field_ext_from(j, 0)));
            }}
            GatesUtilsLib.push(constraints, ev.filter, i, product);
        }}
    }}
}}"
        )
        .to_string();

        template_str = template_str.replace("$NUM_LIMBS", &*self.num_limbs.to_string());
        template_str = template_str.replace("$B", &*B.to_string());

        template_str
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers(&limbs, F::Extension::from_canonical_usize(B));
        let mut constraints = vec![computed_sum - sum];
        for limb in limbs {
            constraints.push(
                (0..B)
                    .map(|i| limb - F::Extension::from_canonical_usize(i))
                    .product(),
            );
        }
        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let base = builder.constant(F::from_canonical_usize(B));
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers_ext_circuit(builder, &limbs, base);
        let mut constraints = vec![builder.sub_extension(computed_sum, sum)];
        for limb in limbs {
            constraints.push({
                let mut acc = builder.one_extension();
                (0..B).for_each(|i| {
                    // We update our accumulator as:
                    // acc' = acc (x - i)
                    //      = acc x + (-i) acc
                    // Since -i is constant, we can do this in one arithmetic_extension call.
                    let neg_i = -F::from_canonical_usize(i);
                    acc = builder.arithmetic_extension(F::ONE, neg_i, acc, limb, acc)
                });
                acc
            });
        }
        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = BaseSplitGenerator::<B> {
            row,
            num_limbs: self.num_limbs,
        };
        vec![Box::new(gen.adapter())]
    }

    // 1 for the sum then `num_limbs` for the limbs.
    fn num_wires(&self) -> usize {
        1 + self.num_limbs
    }

    fn num_constants(&self) -> usize {
        0
    }

    // Bounded by the range-check (x-0)*(x-1)*...*(x-B+1).
    fn degree(&self) -> usize {
        B
    }

    // 1 for checking the sum then `num_limbs` for range-checking the limbs.
    fn num_constraints(&self) -> usize {
        1 + self.num_limbs
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const B: usize> PackedEvaluableBase<F, D>
    for BaseSumGate<B>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires.view(self.limbs());
        let computed_sum = reduce_with_powers(limbs, F::from_canonical_usize(B));

        yield_constr.one(computed_sum - sum);

        let constraints_iter = limbs.iter().map(|&limb| {
            (0..B)
                .map(|i| limb - F::from_canonical_usize(i))
                .product::<P>()
        });
        yield_constr.many(constraints_iter);
    }
}

#[derive(Debug)]
pub struct BaseSplitGenerator<const B: usize> {
    row: usize,
    num_limbs: usize,
}

impl<F: RichField, const B: usize> SimpleGenerator<F> for BaseSplitGenerator<B> {
    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM)]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let sum_value = witness
            .get_target(Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM))
            .to_canonical_u64() as usize;
        debug_assert_eq!(
            (0..self.num_limbs).fold(sum_value, |acc, _| acc / B),
            0,
            "Integer too large to fit in given number of limbs"
        );

        let limbs = (BaseSumGate::<B>::START_LIMBS..BaseSumGate::<B>::START_LIMBS + self.num_limbs)
            .map(|i| Target::wire(self.row, i));
        let limbs_value = (0..self.num_limbs)
            .scan(sum_value, |acc, _| {
                let tmp = *acc % B;
                *acc /= B;
                Some(F::from_canonical_usize(tmp))
            })
            .collect::<Vec<_>>();

        for (b, b_value) in limbs.zip(limbs_value) {
            out_buffer.set_target(b, b_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::base_sum::BaseSumGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(BaseSumGate::<6>::new(11))
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(BaseSumGate::<6>::new(11))
    }
}