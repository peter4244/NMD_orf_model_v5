# Pipeline Dependency Graph

## Data Flow

```
Isopair pipeline (external)
  ├── target_sequences_corrected.fasta
  ├── structures.rds
  └── ORFik scan results
        │
        ▼
data_prep.py ──► results/nmd_orf_data.h5
  (HDF5 dataset: encoded sequences, ORF features, train/val/test splits)
  Also produces:
    results/selected_orfs.tsv
    results/orf_features.tsv
    results/ref_cds_features.tsv
    results/td2_features.tsv
    results/tx_summary.tsv
        │
        ▼
03_train.py ──► results/best_model_{tag}.pt
  (Trained model weights)
        │
        ▼
evaluate.py ──► results/predictions_{tag}.tsv
               results/metrics_{tag}.json
        │
        ├───────────────────────┬──────────────────────────┐
        ▼                       ▼                          ▼
04_interpret_attention.py   deepshap.py                11_kernel_shap_branches.py
  │                           │                          │
  ▼                           ▼                          ▼
  attention_weights_{tag}.tsv deepshap_{branch}_{tag}_   kernel_shap_branch_{tag}.tsv
                              run{1-5}.npz
                              deepshap_joint_{tag}_
                              [orf{1-4}_]run{1}.npz
        │                       │
        │               ┌───────┴───────┐
        │               ▼               ▼
        │    05_interpret_structural.py  07_motif_analysis.py
        │       │                          │
        │       ▼                          ▼
        │    structural_importance_        motif_logo_{branch}_{tag}.tsv
        │    orf_{tag}.tsv                 motif_kmer_{branch}_{tag}.tsv
        │               │
        │               ▼
        │    05_export_sample_importance.py / 05b_export_sample_importance_tsv.py
        │       │
        │       ▼
        │    sample_importance_{tag}.npz / .tsv
        │
        ├───────┴───────────────────────────┐
        ▼                                   ▼
08_export_subgroup_deepshap_tsv.py    06_export_deepshap_tsv.py
  │                                     │
  ▼                                     ▼
  subgroup_kozak_shap_{tag}.tsv       deepshap_{branch}_positional_{tag}_run{N}.tsv
  subgroup_stop_codon_shap_{tag}.tsv
  subgroup_utr5_channel_shap_{tag}.tsv
  subgroup_junction_downstream_shap_{tag}.tsv
  subgroup_frame_periodicity_{tag}.tsv
  subgroup_gc_channel_{tag}.tsv
  motif_logo_{branch}_subgroup_joint_{tag}.tsv    ◄── NEW (joint DeepSHAP)
  shap_profile_{branch}_subgroup_joint_{tag}.tsv  ◄── NEW (joint DeepSHAP)
        │
        ▼
09_export_*.py  (GC content, junction ordinal, poly(A))
  │
  ▼
  gc_content_{tag}.tsv
  junction_ordinal_shap_{tag}.tsv
  polya_sqanti_{tag}.tsv
        │
        ├───────────────────────────────────┐
        ▼                                   ▼
orf_model_report_v5.Rmd               make_architecture_figure.R
  │                                   make_shap_interpretation_figure.R
  ▼
  orf_model_report_v5.html
  figures/*.pdf, figures/*.png
```

## Input Dependencies by Script

| Script | Reads | Produces |
|--------|-------|----------|
| `data_prep.py` | FASTA, structures.rds, ORFik features | `nmd_orf_data.h5`, `selected_orfs.tsv`, `orf_features.tsv`, `ref_cds_features.tsv`, `td2_features.tsv` |
| `03_train.py` | `nmd_orf_data.h5`, `config.yaml` | `best_model_{tag}.pt` |
| `evaluate.py` | `nmd_orf_data.h5`, `best_model_{tag}.pt` | `predictions_{tag}.tsv`, `metrics_{tag}.json` |
| `04_interpret_attention.py` | `nmd_orf_data.h5`, `best_model_{tag}.pt` | `attention_weights_{tag}.tsv` |
| `deepshap.py` | `nmd_orf_data.h5`, `best_model_{tag}.pt` | `deepshap_{branch}_{tag}[_orf{N}]_run{R}.npz` |
| `05_interpret_structural.py` | `nmd_orf_data.h5`, `best_model_{tag}.pt` | `structural_importance_orf_{tag}.tsv` |
| `07_motif_analysis.py` | `deepshap_{atg,stop}_{tag}_run{R}.npz` | `motif_logo_{branch}_{tag}.tsv` |
| `08_export_subgroup_deepshap_tsv.py` | `deepshap_{atg,stop,joint}_{tag}_run{R}.npz`, `ref_cds_features.tsv`, `td2_features.tsv`, `predictions_{tag}.tsv`, `nmd_orf_data.h5` | `subgroup_*.tsv`, `motif_logo_*_subgroup_joint_*.tsv`, `shap_profile_*_subgroup_joint_*.tsv` |
| `11_kernel_shap_branches.py` | `nmd_orf_data.h5`, `best_model_{tag}.pt` | `kernel_shap_branch_{tag}.tsv` |
| `orf_model_report_v5.Rmd` | All TSV/JSON results above | `orf_model_report_v5.html`, `figures/` |

## Subgroup Assignment

Subgroup assignment logic is defined in two places that **must stay synchronized**:

1. **Python:** `08_export_subgroup_deepshap_tsv.py` → `assign_subgroups()`
2. **R:** `orf_model_report_v5.Rmd` → `sec8_data` construction (~line 1745)

Both include TD2 reclassification: transcripts with `category ∈ ATG_LOST_CATS` and `td2_downstream_ejc > 0` are reclassified from "ref ATG lost" to "PTC+".
