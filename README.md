# Nabra (نبرة): Arabic audio deepfake detection

Graduation project on **ArA-DF 2026, Track 1** (`ArabicSpeech/ArA-DF-2026`, revision `2b1f042d`).

**Main model:** XLS-R 300M (frozen) + Wavelet Prompt Tuning (WPT) + AASIST head.
- 1.74 M of 317 M parameters trained (0.55%). Epoch 18, chosen on dev.
- Threshold 10.4609375: the EER threshold on Track 1 dev-test, fixed before the final test.
- Final test (144,210 clips): **EER 5.93%**, AUC 98.59. At the threshold: accuracy 94.15%, FRR 6.08%, FAR 5.77%.

## Notebook

`nabra_full_pipeline.ipynb` holds the whole project, in order:

| Part | Content |
|---|---|
| A | Data analysis, shortcut audit, clean manifests |
| B | WPT training: learning-rate search, WavLM Large vs XLS-R |
| C | Trainable parameters: Head, LoRA r=8, Partial, Full fine-tuning |
| D | Final test |
| E | Error analysis |
| F | GATR explainability |
| G | Golden data |

Training parts are skipped by default and their saved results are shown. EERs and CIs are computed from the saved scores.

Environment: `requirements.txt` (Python 3.13, torch 2.14.1 + CUDA 12.6, transformers 5.18.0).

## Main results (T1 dev-test, whole clip, EER %)

| Method | Trainable % | Dev-test EER [95% CI] | Final-test EER |
|---|---|---|---|
| Head | 0.14 | 6.49 [6.00, 6.94] | 6.59 |
| **WPT** | **0.55** | **6.06 [5.64, 6.46]** | **5.93** |
| LoRA r=8 | 0.64 | 13.70 [13.15, 14.27] | 13.68 |
| Partial (6 layers) | 24.07 | 5.95 [5.53, 6.36] | 5.92 |
| Full FT (lr 1e-6) | 100 | 16.51 [15.80, 17.26] | 16.73 |

- WavLM Large + WPT: dev-test EER 6.44.
- Clips up to 4 s are the hardest (EER 11.67%). Duration alone does not predict the label (EER 46.9%).
- GATR beats a random order on both perturbation tests, on dev and test. The sanity check fails narrowly (0.508; the pass mark is below 0.5).
- Golden data: RMA 40.0% spliced vs 28.2% control (chance 27.9%); RRA 57.7% vs 28.5%.

## Folders

- `artifacts/aradf_eda/`: clean manifests and data-analysis tables.
- `artifacts/results/<run>/`: config, metrics, training log, scores and CodeCarbon emissions of each run.
  - The main run also has `error_analysis/`, `xai_gatr/` and `xai_golden/`.
- `artifacts/results/summary/`: comparison tables and bootstrap CIs; `report_tables/` holds the report tables.
- `report_figures/`: figures used in the report.
- `full_pipeline_outputs/`: tables and figures written by the notebook when it runs.
- `streamlit_app/`: the prototype (detection + GATR explanation). See `streamlit_app/README.md`.

## Not included

- Checkpoints (`best.pt`, about 1.3 GB each), kept on Google Drive.
- Audio: from Hugging Face (`ArabicSpeech/ArA-DF-2026`).
