# nanochat training report

Generated: 2025-10-14 17:25:34

## Environment

### Git Information
- Branch: master
- Commit: dd6ff9a (dirty)
- Message: fix bug in fallback case of find_largest_model

### Hardware
- Platform: Linux
- CPUs: 96 cores (192 logical)
- Memory: 2015.5 GB
- GPUs: 8x NVIDIA H20
- GPU Memory: 761.8 GB total
- CUDA Version: 12.8
- Hourly Rate: $16.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 163,624
- Lines: 3,896
- Files: 23
- Tokens (approx): 40,906
- Dependencies (uv.lock lines): 2,004

Run started: 2025-10-14 17:25:35

---

## Tokenizer training
timestamp: 2025-10-14 17:28:27

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 82.5896
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9151
- token_bytes_std: 2.8736


## Tokenizer evaluation
timestamp: 2025-10-14 17:28:30

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 721 | 1.24 | +3.2% |
| code | 1259 | 576 | 2.19 | 493 | 2.55 | +14.4% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 225 | 4.94 | +13.5% |
| fwe-train | 4208518 | 900364 | 4.67 | 856901 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010356 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 721 | 1.24 | -98.1% |
| code | 1259 | 309 | 4.07 | 493 | 2.55 | -59.5% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 225 | 4.94 | +9.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 856901 | 4.91 | +2.0% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010356 | 4.86 | +1.9% |


## Base model training
timestamp: 2025-10-15 04:31:53

- run: dummy
- depth: 20
- max_seq_len: 2048
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- weight_decay: 0.0000
- matrix_lr: 0.0200
- grad_clip: 1.0000
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- model_tag: 
- Number of parameters: 560,988,160
- Number of FLOPs per token: 3.491758e+09
- Calculated number of iterations: 21,400
- Number of training tokens: 11,219,763,200
- Tokens : Params ratio: 20.0000
- DDP world size: 8
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8112
- Final validation bpb: 0.8112
- CORE metric estimate: 0.2224
- MFU %: 13.00%
- Total training flops: 3.917670e+19
- Total training time: 633.95m
- Peak memory usage: 75422.02MiB


## Base model loss
timestamp: 2025-10-15 04:32:52

- train bpb: 0.8140
- val bpb: 0.8113
- sample 0: <|bos|>The capital of France is Paris. It is the largest city in France and the second largest city in Europe
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile, and malleable metal. It
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Monday. If today is Monday, then tomorrow will be Tuesday. If today is
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is cold. The opposite of cold is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
- sample 5: <|bos|>My favorite color is red. I love the color red. I love the color red. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is the number of times that 5 is divided by 3. If 5


## Base model evaluation
timestamp: 2025-10-15 04:38:51

- Model: base_model (step 21400)
- CORE metric: 0.2101
- hellaswag_zeroshot: 0.2611
- jeopardy: 0.1034
- bigbench_qa_wikidata: 0.5297
- arc_easy: 0.5382
- arc_challenge: 0.1217
- copa: 0.4000
- commonsense_qa: 0.0192
- piqa: 0.3645
- openbook_qa: 0.1173
- lambada_openai: 0.3776
- hellaswag: 0.2584
- winograd: 0.2967
- winogrande: 0.0813
- bigbench_dyck_languages: 0.1050
- agi_eval_lsat_ar: 0.0761
- bigbench_cs_algorithms: 0.3727
- bigbench_operators: 0.1714
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.2237
- coqa: 0.2059
- boolq: -0.1822
- bigbench_language_identification: 0.1792


## Midtraining
timestamp: 2025-10-15 11:09:35

- run: dummy
- dtype: bfloat16
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- final_lr_frac: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- Number of iterations: 765
- DDP world size: 8
- Minimum validation bpb: 0.4149


## Chat evaluation mid
timestamp: 2025-10-15 11:21:02

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- ARC-Easy: 0.3649
- ARC-Challenge: 0.2705
- MMLU: 0.3026
- GSM8K: 0.0318
- HumanEval: 0.0793
- ChatCORE metric: 0.0723


## Chat SFT
timestamp: 2025-10-15 11:42:09

- run: dummy
- source: mid
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- max_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- Training rows: 20,843
- Number of iterations: 651
- Training loss: 1.2072
- Validation loss: 1.0648


## Chat evaluation sft
timestamp: 2025-10-15 11:50:52

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- ARC-Easy: 0.3843
- ARC-Challenge: 0.2884
- MMLU: 0.3095
- GSM8K: 0.0440
- HumanEval: 0.0854
- ChatCORE metric: 0.0878


## Chat RL
timestamp: 2025-10-15 15:15:06

- run: dummy
- source: sft
- dtype: bfloat16
- device_batch_size: 8
- examples_per_step: 16
- num_samples: 16
- max_new_tokens: 256
- temperature: 1.0000
- top_k: 50
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0500
- num_epochs: 1
- save_every: 60
- eval_every: 60
- eval_examples: 400


## Chat evaluation rl
timestamp: 2025-10-15 15:24:36

- source: rl
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- ARC-Easy: 0.3775
- ARC-Challenge: 0.2867
- MMLU: 0.3093
- GSM8K: 0.0788
- HumanEval: 0.0610
- ChatCORE metric: 0.0876


## Summary

- Characters: 163,624
- Lines: 3,896
- Files: 23
- Tokens (approx): 40,906
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2101   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2705   | 0.2884   | -        |
| ARC-Easy        | -        | 0.3649   | 0.3843   | -        |
| GSM8K           | -        | 0.0318   | 0.0440   | 0.0788   |
| HumanEval       | -        | 0.0793   | 0.0854   | -        |
| MMLU            | -        | 0.3026   | 0.3095   | -        |
| ChatCORE        | -        | 0.0723   | 0.0878   | -        |

Total wall clock time: 18h25m
