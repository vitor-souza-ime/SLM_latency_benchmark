# Automated report — GPU-local vs remote SLM benchmark

Observations: 240; valid: 240; failed: 0.

## Per-model, environment, and task-type metrics

                model environment   task_type  n_success  n_questions  latency_p95_s  mean_output_tokens  reasoning_present_pct  e2e_s_median  e2e_s_ci_lo  e2e_s_ci_hi  ttfc_s_median  ttfc_s_ci_lo  ttfc_s_ci_hi  throughput_after_ttfc_tps_median  throughput_after_ttfc_tps_ci_lo  throughput_after_ttfc_tps_ci_hi  accuracy_pct  accuracy_ci_lo  accuracy_ci_hi
Llama 3.1 8B Instruct       local    accuracy         30            6          0.056               2.000                    0.0         0.046        0.038        0.048          0.034         0.026         0.036                            82.639                           82.252                           84.343       100.000         100.000         100.000
Llama 3.1 8B Instruct       local performance         30            6          2.552             139.167                    0.0         1.427        0.811        2.460          0.036         0.033         0.042                            89.978                           89.754                           90.163           NaN             NaN             NaN
Llama 3.1 8B Instruct      remote    accuracy         30            6          1.680               3.300                    0.0         0.696        0.602        0.788          0.689         0.593         0.778                           117.660                           64.562                          279.123        76.667          53.333          96.667
Llama 3.1 8B Instruct      remote performance         30            6          7.287             149.567                    0.0         3.462        1.205        5.874          0.716         0.661         0.898                            59.026                           38.245                          139.427           NaN             NaN             NaN
       Ministral 3 8B       local    accuracy         30            6          0.076               2.167                    0.0         0.049        0.046        0.061          0.036         0.033         0.041                            77.626                           75.968                           78.666       100.000         100.000         100.000
       Ministral 3 8B       local performance         30            6          3.944             180.700                    0.0         1.887        1.338        3.455          0.036         0.034         0.038                            82.095                           81.302                           82.413           NaN             NaN             NaN
       Ministral 3 8B      remote    accuracy         30            6          2.506               2.000                    0.0         0.644        0.595        0.700          0.627         0.564         0.698                            36.884                           31.685                           39.961       100.000         100.000         100.000
       Ministral 3 8B      remote performance         30            6          7.323             178.833                    0.0         2.347        1.728        4.020          0.664         0.579         0.802                           104.687                           91.183                          109.220           NaN             NaN             NaN

## Paired per-prompt comparison (local vs remote latency)

                model   task_type  n_paired_prompts  remote_speedup_median  speedup_ci_lo  speedup_ci_hi  wilcoxon_p  lat_local_median  lat_remote_median  wilcoxon_p_holm
Llama 3.1 8B Instruct    accuracy                 6                 0.0622         0.0528         0.0802      0.0312            0.0461             0.6961           0.0625
Llama 3.1 8B Instruct performance                 6                 0.5146         0.3459         0.6950      0.0312            1.4270             3.4615           0.0625
       Ministral 3 8B    accuracy                 6                 0.0747         0.0700         0.0996      0.0312            0.0490             0.6439           0.0625
       Ministral 3 8B performance                 6                 0.7975         0.7610         0.8788      0.0312            1.8865             2.3475           0.0625

## Interpretation and limitations

- TTFC measures first visible text, not necessarily the first internally generated token; hidden reasoning can precede visible output.
- E2E latency includes networking, gateway and provider queuing, tokenization, prefill, decoding, and transfers; it is not accelerator-only runtime.
- Client-estimated tokens/s after TTFC can be biased by streamed chunk boundaries and hidden reasoning tokens.
- Local quantization and remote backend configuration can differ; pin and document remote providers.
- QA accuracy uses strict A–D letter matching; verbose or malformed answers count as incorrect.
- Bootstrap confidence intervals resample prompts, not within-prompt repetitions as independent examples.
- Report truncations, caching, provider changes, and HTTP 429 errors; rerun affected sessions as needed.
- Do not generalize these findings to all SLMs or all cloud providers.
