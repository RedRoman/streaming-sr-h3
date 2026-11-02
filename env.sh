# streaming_sr0 runtime on 30778.
# The CANN tree under $H3_BASE_ENV is complete, but its own set_env.sh hardcodes the path it was installed at,
# so we export the same variables here against the actual location instead of sourcing it.
export SSR0=/cache/zihaohe/streaming_sr0
export H3_BASE_ENV=/cache/zihaohe/envs/minimax-h3-npu
export CANN="$H3_BASE_ENV/Ascend/cann-9.0.1"
export ASCEND_HOME_PATH="$CANN"
export ASCEND_TOOLKIT_HOME="$CANN"
export ASCEND_OPP_PATH="$CANN/opp"
export ASCEND_AICPU_PATH="$CANN"
export TOOLCHAIN_HOME="$CANN/toolkit"
export PYTHONPATH="$CANN/python/site-packages:$CANN/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export LD_LIBRARY_PATH="$CANN/lib64:$CANN/lib64/plugin/opskernel:$CANN/lib64/plugin/nnengine:$CANN/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH"
export PATH="$CANN/bin:$CANN/compiler/ccec_compiler/bin:$PATH"
ATB="$H3_BASE_ENV/Ascend/nnal/nnal/atb/latest/atb/cxx_abi_1"
export ATB_HOME_PATH="$ATB"
export LD_LIBRARY_PATH="$ATB/lib:$ATB/examples:$LD_LIBRARY_PATH"
export PATH="$ATB/bin:$PATH"
export PATH="$SSR0/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TMPDIR="$SSR0/tmp"
export TORCH_HOME="$SSR0/torch_home"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?set ASCEND_RT_VISIBLE_DEVICES explicitly}"
mkdir -p "$TMPDIR"
