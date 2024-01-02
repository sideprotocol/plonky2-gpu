
# Welcome to the `plonky2-gpu` Repository!

In the pursuit of cryptographic efficiency, we present `plonky2-gpu`—a GPU-accelerated iteration of the [Plonky2](https://github.com/0xPolygonZero/plonky2) project. Leveraging the CUDA framework, this repository embodies a meticulous reengineering of the original Plonky2 codebase, with a specific focus on optimizing three pivotal calculations: Fast Fourier Transform (FFT), Merkle tree construction, and polynomial manipulation.

**Hardware Requirements:**
- **CPU:** 8 cores
- **RAM:** 16GB
- **GPU:** NVIDIA 2080 Ti
- **GPU RAM:** 12GB

The accelerated performance is strikingly evident in real-world scenarios, as exemplified by the reduction of proving time for ed25519 signatures from 45 seconds to an astounding 5 seconds. Developers, cryptographers, and enthusiasts are invited to explore the power of GPU parallelization in cryptographic operations and witness the tangible advancements achieved by `plonky2-gpu`. 

**Examples:**
- [`Plonky2-25519`](https://github.com/sideprotocol/plonky2-ed25519): Experience the optimized performance specifically tailored for ed25519 signatures. 

Join us in the evolution of Zero Knowledge Prove!

## Teams 

 - [Side Labs](https://sidelabs.co)
