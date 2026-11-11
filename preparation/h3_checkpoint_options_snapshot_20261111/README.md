# Checkpoint options snapshot correction

The shared MiniMax `transformer_options` dictionary is now copied at each block call, before later blocks change `block_index`. Each forward/recompute invocation receives its own copy of that snapshot. No model weight, sampling schedule or previous run snapshot changes.

The frozen module is SHA256 `69bc9ca998b02cc9b40704743f63ac14806113732bb250f0c28d7b4bee2fce22`; tests are SHA256 `f9e05c4cadff196e0da0c7ef511321ee21db73e30252dc092df01d1534f61287`. The isolated 30794 CPU execution passed all 10 tests with no skips (4.319 seconds for tests); Torch 2.10.0+cpu, no torch_npu import and zero NPU calls. The exact tool output and structured receipt are retained here.

The new regression makes a shared mutable option affect a nonlinear saved activation, compares output and both input/parameter gradients with the uncheckpointed path, and verifies each block replays its original index. This establishes the wrapper correction on real Torch CPU. It does not establish a prior NPU output error or real H3 forward/backward parity; the strict new latent probe must bind this source and test those separately.
