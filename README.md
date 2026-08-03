# NeuroVis-3D

Official PyTorch implementation of NeuroVis-3D: A Neuromorphic Dual-Stream Network with Dempster-Shafer Evidential Fusion for Reliable 3D Medical Image Classification (IEEE JBHI, 2026).

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python master_neurovis_eval.py
```

This trains and evaluates NeuroVis-3D on four MedMNIST-3D benchmarks.

## Data Preparation

LUNA16 external validation: place dataset in `external_data/` folder.

## Scripts

| Script | Description |
|--------|-------------|
| `master_neurovis_eval.py` | Main evaluation (Table I) |
| `extract_fig2_and_table3.py` | Training curves & failure cases |
| `explain_neurovis_3d.py` | Grad-CAM visualization |
| `generate_paper_figures.py` | Uncertainty density plots |
| `external_validate.py` | LUNA16 transfer learning (Table II) |

## Results

| Dataset | ResNet AUC | NeuroVis-3D AUC | DeLong p |
|---------|------------|-----------------|----------|
| NoduleMNIST3D | 0.8538 | **0.8979** | **0.031** |
| OrganMNIST3D | 0.9981 | **0.9980** | N/A |
| FractureMNIST3D | 0.6843 | 0.5674 | N/A |
| VesselMNIST3D | 0.8117 | **0.9410** | **<0.001** |

## Citation

```bibtex
@article{dengyj2026neurovis,
  title={NeuroVis-3D: A Neuromorphic Dual-Stream Network with Dempster-Shafer Evidential Fusion for Reliable 3D Medical Image Classification}
}
```

## License

MIT
