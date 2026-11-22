# Nabra results so far (ArA-DF 2026, Track 1)

Each folder is a copy of a run folder (all files except `best.pt`). Checkpoints stay in `C:\Users\Aisha\nabra_runs\` (Aisha) and on Dana's Drive.
Verify any run: `python check_scores.py <run_folder>` (recomputes EER from the score files against the clean manifests).

| Run | Model | Method | Best epoch | Dev EER (whole clip) | T1 final-test EER |
|---|---|---|---|---|---|
| `xlsr300m__wpt` (Aisha, RTX 3050 6 GB) | XLS-R 300M | WPT | 8 / 8 | 0.10% | **9.41%** |
| `xlsr300m__fullft` (Dana, Colab L4) | XLS-R 300M | Full FT | 20 / 20 (cap reached, dev still improving) | 3.61% | **16.73%** |
| `wavlm_large__wpt` (Aisha; trained on RTX 3050, test eval on Colab L4) | WavLM-Large | WPT | 7 / 8 | 0.38% | **8.48%** |
| `wavlm_large__fullft` (Dana, Colab L4) | WavLM-Large | Full FT | 19 / 20 (dev flat at 20, lr 1.35e-7, train loss 0.36) | 9.59% | **20.27%** |
| Official baseline (`ArabicSpeech/ArA-DF-Baseline`, results.json) | XLS-R 300M + AASIST | Full FT | up to 100, patience 7 | - | 14.54% |

T1 final-test EER by clip duration:

| Duration | XLS-R WPT | XLS-R Full FT | WavLM WPT | WavLM Full FT |
|---|---|---|---|---|
| < 4.04 s | 15.17% | 21.78% | 13.77% | 20.32% |
| 4.04-8 s | 9.42% | 17.05% | 8.59% | 20.51% |
| 8-12 s | 8.65% | 16.03% | 7.75% | 20.12% |
| 12-20 s | 8.45% | 16.06% | 7.59% | 19.84% |

## Full FT vs official baseline (config.yaml compared line by line)
Same: XLS-R 300M + AASIST, CE weights [0.3, 0.7], Adam lr 1e-6, wd 1e-4, ExponentialLR gamma 0.9, batch 32, seed 42, 64,600 samples repeat-pad, random train crop, no augmentation.
Different: epochs 20/ES5 (team decision) vs 100/ES7; front end HF transformers (input normalization, layerdrop 0.1) vs fairseq `xlsr2_300m.pt`; trim silence + RMS -23 dBFS (EDA section 9) vs none; final eval whole clip vs one start crop.
Note: with gamma 0.9 the summed lr after epoch 20 is only ~12% of the first 20 epochs, so extra epochs alone have a limited effect.

## WavLM WPT note
Trained locally (RTX 3050, 8 epochs, best epoch 7, dev centre crop 0.52%); one accidental interrupt during epoch 8, resumed from last.pt. Training carbon = sum of the two rows in emissions_train.csv: 44,153 s / 0.728 kg. Test evaluation on Colab L4 (5,498 s / 0.104 kg), so its carbon and time are not comparable with the XLS-R WPT evaluation done on the laptop. config.json is the laptop copy (the Colab run rewrites it). A first attempt on a T4 gave dev whole-clip 0.365% vs 0.376% on L4 (bf16 / hardware non-determinism); the L4 run is the one reported.

## WavLM Full FT note
Same baseline recipe as XLS-R (Adam 1e-6, gamma 0.9, 20/ES5). Dev EER fell every epoch to 10.89% (centre crop) at epoch 19 and was flat at 20; train loss still 0.36 when lr had decayed to 1.35e-7. Likely cause (not tested): the baseline lr schedule is tuned for XLS-R and decays before WavLM fits. 21.6 min/epoch on L4. Carbon: train 26,243 s / 0.495 kg, eval 5,185 s / 0.098 kg.

## Published reference numbers (T1 final unless noted)
- UZH-CL (Farhadipour et al.): XLS-R 300M + WPT, 100k Arabic clips, DSU+RawBoost+MUSAN/RIR, cross-layer attention + ASP head: 5.84%. W2V-BERT 2.0 + WPT on official 22.5k: 3.01-9.34%.
- Raid AI (rank 1, 0.87%): XLS-R full FT, same optimiser as baseline; challenge data only: 13.13% (dev-test). Gains came from codec simulation, RIR/noise and extra data.
- PogranichnyZashchitnik: WavLM-Large + LoRA (1.7M params) + ASP + OC-Softmax, with augmentation: 12.54%.
- R_A: WavLM-Large full FT, no augmentation: 6.30% (dev-test); with 50% degradation 2.30% (dev-test).

## Known gaps
- WPT carbon: `emissions_train.csv` covers epochs 7-8 only; epochs 1-6 were lost in a Windows Update reboot.
- Single seed per run.
- Test dialect/gender are "unknown" in the manifest.
