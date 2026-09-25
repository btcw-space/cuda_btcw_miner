# BTCW CUDA Miner — RTX 4060 NEWFORK

CUDA miner for the BTCW `NO_EXT_WORK`/NEWFORK signing-work algorithm. The
default build is tuned for an NVIDIA RTX 4060 (Ada, compute capability 8.9,
8 GB VRAM) and uses BTCW's original 224-byte shared-memory payload.

## Algorithm

For each block template, the BTCW node supplies a 32-byte secp256k1 secret
key, 160 bytes of context, and a fixed 32-byte `hash_no_sig` message.

The GPU searches a 32-bit `test_case`. Each candidate is evaluated as follows:

1. Form RFC6979 extra entropy as `LE32(test_case) || 28 zero bytes`.
2. Generate deterministic ECDSA nonce `k` from the secret key, reduced
   message, and extra entropy. Test case zero is skipped because zero is
   reserved by the result mailbox.
3. Compute `R = kG` on secp256k1 and `r = R.x mod n`.
4. Compute `s = k^-1 * (hash_no_sig + r * secret_key) mod n`.
5. Normalize `s` to low-S form and DER-encode `(r,s)`.
6. Accept only 70- or 71-byte DER signatures.
7. Compute `SHA256(SHA256(DER_signature))`.
8. Compare the result with the fixed Stage-2 target `2^228 - 1`, equivalent
   to 28 leading zero bits in displayed hash order.
9. Return a successful 32-bit `test_case` through the original 64-bit
   shared-memory nonce mailbox (`0x0707070707070707` idle sentinel).

The BTCW node remains the final authority. It regenerates the signature using
`CKey::Sign(hash_no_sig, ..., false, test_case)` and validates it before block
submission.

## CUDA design

Important optimizations include:

- fixed-layout, word-native RFC6979/HMAC-SHA256;
- reuse of secret-key and reduced-message SHA prefix state;
- secp256k1 GLV scalar splitting;
- a shared W24/W9 fixed-base lookup table of approximately 2.5 GiB;
- XYZZ mixed point addition;
- 128-candidate batched scalar inversion;
- two 64-candidate field-inversion sub-batches;
- PTX 32-bit Comba field and scalar multiplication;
- specialized DER SHA256d;
- direct comparison of the final SHA word with the fixed target.

The miner runs an RFC6979 GPU self-test at startup. It compares the optimized
extra-entropy implementation with the generic reference implementation for
known test cases and aborts if they differ.

## Requirements

- Linux x86-64
- NVIDIA driver
- CUDA Toolkit with `nvcc`
- RTX 4060 or another CUDA-capable NVIDIA GPU
- roughly 4 GiB of free GPU memory for the default configuration
- a compatible BTCW NEWFORK node exposing `/shared_mem`

## Build

Make the script executable and build:

```bash
chmod +x build_cuda.sh
./build_cuda.sh
```

The default configuration is:

```text
CUDA architecture: sm_89
register cap:      160
sign batch:        128
```

It creates:

```text
release/btcw_cuda_miner_batch128
release/btcw_cuda_miner
```

Both files contain the same default build. The second name is a compatibility
copy.

Specify a different register cap with the first argument:

```bash
./build_cuda.sh 160
```

Build batch 64 for comparison:

```bash
CUDA_SIGN_BATCH=64 ./build_cuda.sh 160
```

This produces `release/btcw_cuda_miner_batch64` and updates the compatibility
copy `release/btcw_cuda_miner`.

Build for a different CUDA architecture:

```bash
CUDA_ARCH=sm_89 ./build_cuda.sh 160
```

The compiler prints register, stack, and spill statistics for the kernels.
Large local frames are expected because the mining kernel batches many
secp256k1 operations.

## Run

Run the recommended batch-128 binary from the repository root:

```bash
./release/btcw_cuda_miner_batch128
```

Command-line syntax:

```text
./release/btcw_cuda_miner_batch128 [gpu_number] [work_size] [block_size]
```

Examples:

```bash
# Default GPU and automatically selected work size
./release/btcw_cuda_miner_batch128

# Explicit RTX 4060 launch settings
./release/btcw_cuda_miner_batch128 0 233472 128
```

The tuned defaults for a 24-SM RTX 4060 are:

```text
work size:  24 * 9728 = 233472
block size: 128
sign batch: 128
```

At startup, confirm that the output contains:

```text
sign-batch128
RFC6979 extra-entropy fast path self-test passed.
```

Let the miner run through several two-second reporting intervals before
recording hashrate. Do not run multiple miners on the same GPU while testing.

Stop cleanly with `Ctrl+C`.

## Shared-memory interface

The miner communicates through POSIX shared memory named `/shared_mem`:

```text
offset 0:   uint64 nonce/result mailbox
offset 8:   32-byte secret key
offset 40:  160-byte context
offset 200: 32-byte hash_no_sig
```

The node payload is 224 bytes, plus the 8-byte mailbox. The CUDA miner installs
the fixed Stage-2 target locally; target bytes are not read from shared memory.

Only run one BTCW node/miner pair against a given `/shared_mem` object.

## Troubleshooting

Verify the driver and compiler:

```bash
nvidia-smi
nvcc --version
```

If the miner reports that it is not connected, ensure the BTCW node and
wallet are running and that the wallet has at least one UTXO.

If the RFC6979 self-test fails, do not mine with that binary. Rebuild for the
correct CUDA architecture and investigate the compiler configuration.
