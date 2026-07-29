# NeuroVis-3D

A neuromorphic dual-stream network with Dempster-Shafer evidential fusion for reliable 3D medical image classification.

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main evaluation script:

```bash
python master_neurovis_eval.py
```

This will train and evaluate NeuroVis-3D on four MedMNIST-3D benchmarks (Nodule, Organ, Vessel, Fracture).

## Additional Scripts

| Script | Description |
|--------|-------------|
| `ultimate_stats_test.py` | DeLong/Bootstrap/Permutation statistical tests |
| `generate_paper_figures.py` | Generate paper figures |
| `explain_neurovis_3d.py` | Grad-CAM visualization |
| `external_validate.py` | LUNA16 external validation |

## Results

| Dataset | ResNet AUC | NeuroVis-3D AUC | DeLong p |
|---------|------------|-----------------|----------|
| NoduleMNIST3D | 0.8693 | 0.8984 | 0.044 |
| VesselMNIST3D | 0.8489 | 0.8918 | 0.029 |

## Citation

If you find this work useful, please cite our paper (to be added).

## License

MIT
