use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::marker::PhantomData;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::hashing::SPONGE_WIDTH;
use crate::hash::poseidon;
use crate::hash::poseidon::Poseidon;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};

/// Evaluates a full Poseidon permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Debug, Default)]
pub struct PoseidonGate<F: RichField + Extendable<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> PoseidonGate<F, D> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_FULL_0: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize =
        Self::START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < poseidon::N_PARTIAL_ROUNDS);
        Self::START_PARTIAL + round
    }

    const START_FULL_1: usize = Self::START_PARTIAL + poseidon::N_PARTIAL_ROUNDS;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_1 + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    fn end() -> usize {
        Self::START_FULL_1 + SPONGE_WIDTH * poseidon::HALF_N_FULL_ROUNDS
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for PoseidonGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn export_circom_verification_code(&self) -> String {
        let mut template_str = format!(
            "template Poseidon12() {{
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  $SET_FILTER;

  var index = 0;
  out[index] <== ConstraintPush()(constraints[index], filter, GlExtMul()(wires[$WIRE_SWAP], GlExtSub()(wires[$WIRE_SWAP], GlExt(1, 0)())));
  index++;

  for (var i = 0; i < 4; i++) {{
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(GlExtMul()(wires[$WIRE_SWAP], GlExtSub()(wires[i + 4], wires[i])), wires[$START_DELTA + i]));
    index++;
  }}

  // SPONGE_RATE = 8
  // SPONGE_CAPACITY = 4
  // SPONGE_WIDTH = 12
  signal state[12][$HALF_N_FULL_ROUNDS * 8 + 2 + $N_PARTIAL_ROUNDS * 2][2];
  var state_round = 0;
  for (var i = 0; i < 4; i++) {{
    state[i][state_round] <== GlExtAdd()(wires[i], wires[$START_DELTA + i]);
    state[i + 4][state_round] <== GlExtSub()(wires[i + 4], wires[$START_DELTA + i]);
  }}

  for (var i = 8; i < 12; i++) {{
    state[i][state_round] <== wires[i];
  }}
  state_round++;

  var round_ctr = 0;
  // First set of full rounds.
  signal mds_row_shf_field[$HALF_N_FULL_ROUNDS][12][13][2];
  for (var r = 0; r < $HALF_N_FULL_ROUNDS; r ++) {{
    for (var i = 0; i < 12; i++) {{
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(GL_CONST(i + 12 * round_ctr), 0)());
    }}
    state_round++;
    if (r != 0 ) {{
      for (var i = 0; i < 12; i++) {{
        state[i][state_round] <== wires[$START_DELTA + 4 + 12 * (r - 1) + i];
        out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], state[i][state_round]));
        index++;
      }}
      state_round++;
    }}
    for (var i = 0; i < 12; i++) {{
      state[i][state_round] <== GlExtExpN(3)(state[i][state_round - 1], 7);
    }}
    state_round++;
    for (var i = 0; i < 12; i++) {{ // for r
      mds_row_shf_field[r][i][0][0] <== 0;
      mds_row_shf_field[r][i][0][1] <== 0;
      for (var j = 0; j < 12; j++) {{ // for i,
        mds_row_shf_field[r][i][j + 1] <== GlExtAdd()(mds_row_shf_field[r][i][j], GlExtMul()(state[(i + j) < 12 ? (i + j) : (i + j - 12)][state_round - 1], GlExt(MDS_MATRIX_CIRC(j), 0)()));
      }}
      state[i][state_round] <== GlExtAdd()(mds_row_shf_field[r][i][12], GlExtMul()(state[i][state_round - 1], GlExt(MDS_MATRIX_DIAG(i), 0)()));
    }}
    state_round++;
    round_ctr++;
  }}

  // Partial rounds.
  for (var i = 0; i < 12; i++) {{
    state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(FAST_PARTIAL_FIRST_ROUND_CONSTANT(i), 0)());
  }}
  state_round++;
  component partial_res[11][11];
  state[0][state_round] <== state[0][state_round - 1];
  for (var r = 0; r < 11; r++) {{
    for (var c = 0; c < 11; c++) {{
      partial_res[r][c] = GlExtAdd();
      if (r == 0) {{
        partial_res[r][c].a <== GlExt(0, 0)();
      }} else {{
        partial_res[r][c].a <== partial_res[r - 1][c].out;
      }}
      partial_res[r][c].b <== GlExtMul()(state[r + 1][state_round - 1], GlExt(FAST_PARTIAL_ROUND_INITIAL_MATRIX(r, c), 0)());
    }}
  }}
  for (var i = 1; i < 12; i++) {{
    state[i][state_round] <== partial_res[10][i - 1].out;
  }}
  state_round++;

  signal partial_d[12][$N_PARTIAL_ROUNDS][2];
  for (var r = 0; r < $N_PARTIAL_ROUNDS; r++) {{
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[0][state_round - 1], wires[$START_PARTIAL + r]));
    index++;
    if (r == $N_PARTIAL_ROUNDS - 1) {{
      state[0][state_round] <== GlExtExpN(3)(wires[$START_PARTIAL + r], 7);
    }} else {{
      state[0][state_round] <== GlExtAdd()(GlExt(FAST_PARTIAL_ROUND_CONSTANTS(r), 0)(), GlExtExpN(3)(wires[$START_PARTIAL + r], 7));
    }}
    for (var i = 1; i < 12; i++) {{
      state[i][state_round] <== state[i][state_round - 1];
    }}
    partial_d[0][r] <== GlExtMul()(state[0][state_round], GlExt(MDS_MATRIX_CIRC(0) + MDS_MATRIX_DIAG(0), 0)());
    for (var i = 1; i < 12; i++) {{
      partial_d[i][r] <== GlExtAdd()(partial_d[i - 1][r], GlExtMul()(state[i][state_round], GlExt(FAST_PARTIAL_ROUND_W_HATS(r, i - 1), 0)()));
    }}
    state_round++;
    state[0][state_round] <== partial_d[11][r];
    for (var i = 1; i < 12; i++) {{
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExtMul()(state[0][state_round - 1], GlExt(FAST_PARTIAL_ROUND_VS(r, i - 1), 0)()));
    }}
    state_round++;
  }}
  round_ctr += $N_PARTIAL_ROUNDS;

  // Second set of full rounds.
  signal mds_row_shf_field2[$HALF_N_FULL_ROUNDS][12][13][2];
  for (var r = 0; r < $HALF_N_FULL_ROUNDS; r ++) {{
    for (var i = 0; i < 12; i++) {{
      state[i][state_round] <== GlExtAdd()(state[i][state_round - 1], GlExt(GL_CONST(i + 12 * round_ctr), 0)());
    }}
    state_round++;
    for (var i = 0; i < 12; i++) {{
      state[i][state_round] <== wires[$START_FULL_1 + 12 * r + i];
      out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], state[i][state_round]));
      index++;
    }}
    state_round++;
    for (var i = 0; i < 12; i++) {{
      state[i][state_round] <== GlExtExpN(3)(state[i][state_round - 1], 7);
    }}
    state_round++;
    for (var i = 0; i < 12; i++) {{ // for r
      mds_row_shf_field2[r][i][0][0] <== 0;
      mds_row_shf_field2[r][i][0][1] <== 0;
      for (var j = 0; j < 12; j++) {{ // for i,
        mds_row_shf_field2[r][i][j + 1] <== GlExtAdd()(mds_row_shf_field2[r][i][j], GlExtMul()(state[(i + j) < 12 ? (i + j) : (i + j - 12)][state_round - 1], GlExt(MDS_MATRIX_CIRC(j), 0)()));
      }}
      state[i][state_round] <== GlExtAdd()(mds_row_shf_field2[r][i][12], GlExtMul()(state[i][state_round - 1], GlExt(MDS_MATRIX_DIAG(i), 0)()));
    }}
    state_round++;
    round_ctr++;
  }}

  for (var i = 0; i < 12; i++) {{
    out[index] <== ConstraintPush()(constraints[index], filter, GlExtSub()(state[i][state_round - 1], wires[12 + i]));
    index++;
  }}

  for (var i = index + 1; i < NUM_GATE_CONSTRAINTS(); i++) {{
    out[i] <== constraints[i];
  }}
}}
function FAST_PARTIAL_ROUND_W_HATS(i, j) {{
  var value[$N_PARTIAL_ROUNDS][11];
  $SET_FAST_PARTIAL_ROUND_W_HATS;
  return value[i][j];
}}
function FAST_PARTIAL_ROUND_VS(i, j) {{
  var value[$N_PARTIAL_ROUNDS][11];
  $SET_FAST_PARTIAL_ROUND_VS;
  return value[i][j];
}}
function FAST_PARTIAL_ROUND_INITIAL_MATRIX(i, j) {{
  var value[11][11];
  $SET_FAST_PARTIAL_ROUND_INITIAL_MATRIX;
  return value[i][j];
}}
function FAST_PARTIAL_ROUND_CONSTANTS(i) {{
  var value[$N_PARTIAL_ROUNDS];
  $SET_FAST_PARTIAL_ROUND_CONSTANTS;
  return value[i];
}}
function FAST_PARTIAL_FIRST_ROUND_CONSTANT(i) {{
  var value[12];
  $SET_FAST_PARTIAL_FIRST_ROUND_CONSTANT;
  return value[i];
}}
function MDS_MATRIX_CIRC(i) {{
  var mds[12];
  $SET_MDS_MATRIX_CIRC;
  return mds[i];
}}
function MDS_MATRIX_DIAG(i) {{
  var mds[12];
  $SET_MDS_MATRIX_DIAG;
  return mds[i];
}}"
        ).to_string();
        template_str = template_str.replace("$WIRE_SWAP", &*Self::WIRE_SWAP.to_string());
        template_str = template_str.replace("$START_DELTA", &*Self::START_DELTA.to_string());
        template_str = template_str.replace("$START_FULL_1", &*Self::START_FULL_1.to_string());
        template_str = template_str.replace(
            "$HALF_N_FULL_ROUNDS",
            &*poseidon::HALF_N_FULL_ROUNDS.to_string(),
        );
        template_str = template_str.replace(
            "$N_PARTIAL_ROUNDS",
            &*poseidon::N_PARTIAL_ROUNDS.to_string(),
        );
        template_str = template_str.replace("$START_PARTIAL", &*Self::START_PARTIAL.to_string());

        let mut partial_const_str = "".to_owned();
        for i in 0..poseidon::N_PARTIAL_ROUNDS {
            partial_const_str += &*("  value[".to_owned()
                + &*i.to_string()
                + "] = "
                + &*<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[i].to_string()
                + ";\n");
        }
        template_str = template_str.replace(
            "  $SET_FAST_PARTIAL_ROUND_CONSTANTS;\n",
            &*partial_const_str,
        );

        let mut circ_str = "".to_owned();
        for i in 0..12 {
            circ_str += &*("  mds[".to_owned()
                + &*i.to_string()
                + "] = "
                + &*<F as Poseidon>::MDS_MATRIX_CIRC[i].to_string()
                + ";\n");
        }
        template_str = template_str.replace("  $SET_MDS_MATRIX_CIRC;\n", &*circ_str);

        let mut diag_str = "".to_owned();
        for i in 0..12 {
            diag_str += &*("  mds[".to_owned()
                + &*i.to_string()
                + "] = "
                + &*<F as Poseidon>::MDS_MATRIX_DIAG[i].to_string()
                + ";\n");
        }
        template_str = template_str.replace("  $SET_MDS_MATRIX_DIAG;\n", &*diag_str);

        let mut first_round_const_str = "".to_owned();
        for i in 0..12 {
            first_round_const_str += &*("  value[".to_owned()
                + &*i.to_string()
                + "] = "
                + &*<F as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i].to_string()
                + ";\n");
        }
        template_str = template_str.replace(
            "  $SET_FAST_PARTIAL_FIRST_ROUND_CONSTANT;\n",
            &*first_round_const_str,
        );

        let mut init_m_str = "".to_owned();
        for i in 0..11 {
            for j in 0..11 {
                init_m_str += &*("  value[".to_owned()
                    + &*i.to_string()
                    + "]["
                    + &*j.to_string()
                    + "] = "
                    + &*<F as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[i][j].to_string()
                    + ";\n");
            }
        }
        template_str =
            template_str.replace("  $SET_FAST_PARTIAL_ROUND_INITIAL_MATRIX;\n", &*init_m_str);

        let mut partial_hats_str = "".to_owned();
        for i in 0..poseidon::N_PARTIAL_ROUNDS {
            for j in 0..11 {
                partial_hats_str += &*("  value[".to_owned()
                    + &*i.to_string()
                    + "]["
                    + &*j.to_string()
                    + "] = "
                    + &*<F as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[i][j].to_string()
                    + ";\n");
            }
        }
        template_str =
            template_str.replace("  $SET_FAST_PARTIAL_ROUND_W_HATS;\n", &*partial_hats_str);

        let mut partial_vs_str = "".to_owned();
        for i in 0..poseidon::N_PARTIAL_ROUNDS {
            for j in 0..11 {
                partial_vs_str += &*("  value[".to_owned()
                    + &*i.to_string()
                    + "]["
                    + &*j.to_string()
                    + "] = "
                    + &*<F as Poseidon>::FAST_PARTIAL_ROUND_VS[i][j].to_string()
                    + ";\n");
            }
        }
        template_str = template_str.replace("  $SET_FAST_PARTIAL_ROUND_VS;\n", &*partial_vs_str);

        template_str
    }
    fn export_solidity_verification_code(&self) -> String {
        todo!()
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
            state[0] +=
                F::Extension::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast_field(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        constraints.push(state[0] - sbox_in);
        state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
        state =
            <F as Poseidon>::mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
            state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        yield_constr.one(state[0] - sbox_in);
        state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
        state = <F as Poseidon>::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        // The naive method is more efficient if we have enough routed wires for PoseidonMdsGate.
        let use_mds_gate =
            builder.config.num_routed_wires >= PoseidonMdsGate::<F, D>::new().num_wires();

        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        // Partial rounds.
        if use_mds_gate {
            for r in 0..poseidon::N_PARTIAL_ROUNDS {
                <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
                state = <F as Poseidon>::mds_layer_circuit(builder, &state);
                round_ctr += 1;
            }
        } else {
            <F as Poseidon>::partial_first_constant_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_partial_layer_init_circuit(builder, &state);
            for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
                let c = <F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r];
                let c = F::Extension::from_canonical_u64(c);
                let c = builder.constant_extension(c);
                state[0] = builder.add_extension(state[0], c);
                state = <F as Poseidon>::mds_partial_layer_fast_circuit(builder, &state, r);
            }
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
            state = <F as Poseidon>::mds_partial_layer_fast_circuit(
                builder,
                &state,
                poseidon::N_PARTIAL_ROUNDS - 1,
            );
            round_ctr += poseidon::N_PARTIAL_ROUNDS;
        }

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = PoseidonGenerator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![Box::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * (poseidon::N_FULL_ROUNDS_TOTAL - 1)
            + poseidon::N_PARTIAL_ROUNDS
            + SPONGE_WIDTH
            + 1
            + 4
    }
}

#[derive(Debug)]
struct PoseidonGenerator<F: RichField + Extendable<D> + Poseidon, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> SimpleGenerator<F>
    for PoseidonGenerator<F, D>
{
    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| PoseidonGate::<F, D>::wire_input(i))
            .chain(Some(PoseidonGate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| witness.get_wire(local_wire(PoseidonGate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(PoseidonGate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_delta(i)), delta_i);
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();
        let mut round_ctr = 0;

        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(PoseidonGate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    );
                }
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            out_buffer.set_wire(
                local_wire(PoseidonGate::<F, D>::wire_partial_sbox(r)),
                state[0],
            );
            state[0] = <F as Poseidon>::sbox_monomial(state[0]);
            state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast_field(&state, r);
        }
        out_buffer.set_wire(
            local_wire(PoseidonGate::<F, D>::wire_partial_sbox(
                poseidon::N_PARTIAL_ROUNDS - 1,
            )),
            state[0],
        );
        state[0] = <F as Poseidon>::sbox_monomial(state[0]);
        state =
            <F as Poseidon>::mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(PoseidonGate::<F, D>::wire_full_sbox_1(r, i)),
                    state[i],
                );
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_output(i)), state[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon::PoseidonGate;
    use crate::hash::hashing::SPONGE_WIDTH;
    use crate::hash::poseidon::Poseidon;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::wire::Wire;
    use crate::iop::witness::{PartialWitness, Witness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn wire_indices() {
        type F = GoldilocksField;
        type Gate = PoseidonGate<F, 4>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_full_sbox_0(1, 0), 29);
        assert_eq!(Gate::wire_full_sbox_0(3, 0), 53);
        assert_eq!(Gate::wire_full_sbox_0(3, 11), 64);
        assert_eq!(Gate::wire_partial_sbox(0), 65);
        assert_eq!(Gate::wire_partial_sbox(21), 86);
        assert_eq!(Gate::wire_full_sbox_1(0, 0), 87);
        assert_eq!(Gate::wire_full_sbox_1(3, 0), 123);
        assert_eq!(Gate::wire_full_sbox_1(3, 11), 134);
    }

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = PoseidonGate<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..SPONGE_WIDTH)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs.set_wire(
            Wire {
                row,
                column: Gate::WIRE_SWAP,
            },
            F::ZERO,
        );
        for i in 0..SPONGE_WIDTH {
            inputs.set_wire(
                Wire {
                    row,
                    column: Gate::wire_input(i),
                },
                permutation_inputs[i],
            );
        }

        let witness = generate_partial_witness(inputs, &circuit.prover_only, &circuit.common);

        let expected_outputs: [F; SPONGE_WIDTH] =
            F::poseidon(permutation_inputs.try_into().unwrap());
        for i in 0..SPONGE_WIDTH {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = PoseidonGate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonGate::<F, 2>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
