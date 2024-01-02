
struct ComparisonGate  INHERIT_BASE {
    usize num_bits;
    usize num_chunks;

    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        return 6 + 5 * this->num_chunks + this->chunk_bits();
    }
    __device__ inline
    usize chunk_bits() const {
	return ceil_div_usize(this->num_bits, this->num_chunks);
    }

    __device__ inline
    usize wire_first_input() {
	return 0;
    }

    __device__ inline
    usize wire_second_input() {
	return 1;
    }

    __device__ inline
    usize wire_result_bool() {
	return 2;
    }

    __device__ inline
    usize wire_most_significant_diff() {
	return 3;
    }

    __device__ inline
    usize wire_first_chunk_val(usize chunk) {
        assert(chunk < this->num_chunks);
	return 4 + chunk;
    }

    __device__ inline
    usize wire_second_chunk_val(usize chunk) {
        assert(chunk < this->num_chunks);
	return 4 + this->num_chunks + chunk;
    }

    __device__ inline
    usize wire_equality_dummy(usize chunk) {
        assert(chunk < this->num_chunks);
	return 4 + 2 * this->num_chunks + chunk;
    }

    __device__ inline
    usize wire_chunks_equal(usize chunk) {
        assert(chunk < this->num_chunks);
	return 4 + 3 * this->num_chunks + chunk;
    }

    __device__ inline
    usize wire_intermediate_value(usize chunk) {
        assert(chunk < this->num_chunks);
	return 4 + 4 * this->num_chunks + chunk;
    }

    __device__ inline
/// The `bit_index`th bit of 2^n - 1 + most_significant_diff.
    usize wire_most_significant_diff_bit(usize bit_index) {
	return 4 + 5 * this->num_chunks + bit_index;
    }


    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        auto first_input = vars.local_wires[this->wire_first_input()];
        auto second_input = vars.local_wires[this->wire_second_input()];

        // Get chunks and assert that they match
        auto first_chunks = [this, vars](int i) -> GoldilocksField {
            return vars.local_wires[this->wire_first_chunk_val(i)];
        };

        auto second_chunks = [this, vars](int i) -> GoldilocksField {
            return vars.local_wires[this->wire_second_chunk_val(i)];
        };

        auto first_chunks_combined = reduce_with_powers(
                Range<int>{0, this->num_chunks}, first_chunks, GoldilocksField::from_canonical_usize(1 << this->chunk_bits()));
        auto second_chunks_combined = reduce_with_powers(
                Range<int>{0, this->num_chunks}, second_chunks, GoldilocksField::from_canonical_usize(1 << this->chunk_bits()));

        yield_constr.one(first_chunks_combined - first_input);
        yield_constr.one(second_chunks_combined - second_input);

        auto chunk_size = 1 << this->chunk_bits();

        auto most_significant_diff_so_far = GoldilocksField{0};

        for (int i = 0; i < this->num_chunks; ++i) {
            // Range-check the chunks to be less than `chunk_size`.
            GoldilocksField first_product = GoldilocksField{1};
            for (int j = 0; j < chunk_size; ++j) {
                first_product *= first_chunks(i) - GoldilocksField::from_canonical_usize(j);
            }

            GoldilocksField second_product = GoldilocksField{1};
            for (int j = 0; j < chunk_size; ++j) {
                second_product *= second_chunks(i) - GoldilocksField::from_canonical_usize(j);
            }

            yield_constr.one(first_product);
            yield_constr.one(second_product);

            auto difference = second_chunks(i) - first_chunks(i);
            auto equality_dummy = vars.local_wires[this->wire_equality_dummy(i)];
            auto chunks_equal = vars.local_wires[this->wire_chunks_equal(i)];

            // Two constraints to assert that `chunks_equal` is valid.
            yield_constr.one(difference * equality_dummy - (GoldilocksField{1} - chunks_equal));
            yield_constr.one(chunks_equal * difference);

            // Update `most_significant_diff_so_far`.
            auto intermediate_value = vars.local_wires[this->wire_intermediate_value(i)];
            yield_constr.one(intermediate_value - chunks_equal * most_significant_diff_so_far);
            most_significant_diff_so_far =
                    intermediate_value + (GoldilocksField{1} - chunks_equal) * difference;
        }

        auto most_significant_diff = vars.local_wires[this->wire_most_significant_diff()];
        yield_constr.one(most_significant_diff - most_significant_diff_so_far);

        auto most_significant_diff_bits = [vars, this](int i) -> GoldilocksField {
            return vars.local_wires[this->wire_most_significant_diff_bit(i)];
        };

        // Range-check the bits.
        for (int i = 0; i < this->chunk_bits()+1; ++i) {
            auto bit = most_significant_diff_bits(i);
            yield_constr.one(bit * (GoldilocksField{1} - bit));
        }

        auto bits_combined = reduce_with_powers(Range<int>{0, this->chunk_bits() + 1},  most_significant_diff_bits, GoldilocksField{2});
        auto two_n = GoldilocksField::from_canonical_u64(1 << this->chunk_bits());
        yield_constr.one((most_significant_diff + two_n) - bits_combined);

        // Iff first <= second, the top (n + 1st) bit of (2^n - 1 + most_significant_diff) will be 1.
        auto result_bool = vars.local_wires[this->wire_result_bool()];
        yield_constr.one(result_bool - most_significant_diff_bits(this->chunk_bits()));
    }
};

