struct RandomAccessGate INHERIT_BASE {
/// Number of bits in the index (log2 of the list size).
    usize bits;

/// How many separate copies are packed into one gate.
    usize num_copies;

/// Leftover wires are used as global scratch space to store constants.
    usize num_extra_constants;


    __device__ inline
    VIRTUAL int num_constraints() const OVERRIDE {
        auto constraints_per_copy = this->bits + 2;
        return this->num_copies * constraints_per_copy + this->num_extra_constants;
    }

    __device__ inline
    VIRTUAL void eval_unfiltered_base_packed(
            EvaluationVarsBasePacked vars,
            StridedConstraintConsumer yield_constr) OVERRIDE {
        for (int copy = 0; copy < num_copies; ++copy) {
            auto access_index = vars.local_wires[this->wire_access_index(copy)];
            auto claimed_element = vars.local_wires[this->wire_claimed_element(copy)];

            auto bits = [vars, this, copy](int i) -> GoldilocksField {
                return vars.local_wires[this->wire_bit(i, copy)];
            };
            auto bitsRange = Range<int>{0, this->bits};


            // Assert that each bit wire value is indeed boolean.
            for (auto i: bitsRange) {
                auto b = bits(i);
                yield_constr.one(b * (b - GoldilocksField{1}));
            }

            // Assert that the binary decomposition was correct.
            GoldilocksField reconstructed_index = {0};
            for (int i = bitsRange.second-1; i >= bitsRange.first; --i) {
                auto b = bits(i);
                reconstructed_index += reconstructed_index + b;
            }
            yield_constr.one(reconstructed_index - access_index);

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.

            assert(this->vec_size() % 2 == 0);
            GoldilocksField list_items[235];
            assert(this->vec_size() < sizeof(list_items)/sizeof(GoldilocksField));
            for (int i = 0; i < this->vec_size(); ++i) {
                list_items[i] = vars.local_wires[this->wire_list_item(i, copy)];
            }
            int count = this->vec_size();
            for (auto i: bitsRange) {
                auto b = bits(i);

                assert(count % 2 == 0);
                int c = 0;
                for (int j = 0; j < count; j+=2) {
                    auto x = list_items[j];
                    auto y = list_items[j+1];
                    list_items[c++] = x + b * (y - x);
                }
                count = c;
            }

//            assert(list_items.len(), 1);
            yield_constr.one(list_items[0] - claimed_element);
        }
        for (int i = 0; i < this->num_extra_constants; ++i) {
            yield_constr.one(vars.local_constants[i] - vars.local_wires[this->wire_extra_constant(i)]);
        }
    }




    /// Length of the list being accessed.
    __device__ inline
    usize vec_size() {
	return 1 << this->bits;
    }

    /// For each copy, a wire containing the claimed index of the element.
    __device__ inline
    usize wire_access_index(usize copy) {
        assert(copy < this->num_copies);
        return (2 + this->vec_size()) * copy; 
    }

    /// For each copy, a wire containing the element claimed to be at the index.
    __device__ inline
    usize wire_claimed_element(usize copy) {
        assert(copy < this->num_copies);
        return (2 + this->vec_size()) * copy + 1; 
    }

    /// For each copy, wires containing the entire list.
    __device__ inline
    usize wire_list_item(usize i, usize copy) {
        assert(i < this->vec_size());
        assert(copy < this->num_copies);
        return (2 + this->vec_size()) * copy + 2 + i; 
    }

    __device__ inline
    usize start_extra_constants() {
        return (2 + this->vec_size()) * this->num_copies; 
    }

    __device__ inline
    usize wire_extra_constant(usize i) {
        assert(i < this->num_extra_constants);
        return this->start_extra_constants() + i; 
    }

    /// All above wires are routed.
    __device__ inline
    usize num_routed_wires() {
        return this->start_extra_constants() + this->num_extra_constants; 
    }

    /// An intermediate wire where the prover gives the (purported) binary decomposition of the
    /// index.
    __device__ inline
    usize wire_bit(usize i, usize copy) {
        assert(i < this->bits);
        assert(copy < this->num_copies);
        return this->num_routed_wires() + copy * this->bits + i; 
    }
};

