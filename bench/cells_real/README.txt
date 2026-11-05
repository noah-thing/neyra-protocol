Whole-file cell test on public models; see ../cells_protocol.md for the frozen protocol and ../../docs/source.md for results.

  python -m bench.cells_real.whole_file unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf Qwen/Qwen3-0.6B base=Qwen/Qwen3-0.6B-Base
  python -m bench.cells_real.controls Q4_K [requant]      # compiles ggml-quants.c at a pinned llama.cpp commit
  python -m bench.cells_real.gptq Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-0.5B-Instruct gptq

GGUF tensors are read by HTTP range and never stored. About 3 GB of memory, 7 minutes per file.
Run from the repository root with PYTHONPATH=src.
