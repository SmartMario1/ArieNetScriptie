# ArieNet with Backbone Prediction

This combines ArieNet's GNN architecture with backbone prediction objective.

## Setup

The model uses:
- **BPG format** from nsnet with local satisfaction percentages
- **Backbone labels** computed for SAT instances
- **3-SAT data** from nsnet's SATSolving directory

## Workflow

### 1. Generate Backbone Labels

First, generate backbone labels for your training and validation data:

```bash
# Generate backbone labels for training set
python generate_backbone_labels.py SATSolving/3-sat/train_first_4000_ArieNet --n_process 4

# Generate backbone labels for validation set  
python generate_backbone_labels.py SATSolving/3-sat/valid_first_1000_ArieNet --n_process 4
```

This will create:
- `backbones.pkl` - a pickle file mapping CNF paths to backbone dictionaries
- `backbone_files/` - individual .backbone files (optional, for debugging)

**Requirements**: Install PySAT for backbone computation:
```bash
pip install python-sat
```

### 2. Train the Model

After generating backbone labels, train the model:

```bash
# Pretraining
python train_arienet_backbone.py pretrain

# Finetuning (loads pretrained model)
python train_arienet_backbone.py finetune
```

## Key Features

### ArieNet Architecture
✅ Uses full BPG format with:
- Pre-computed message passing indices (c2l, l2c)
- Local satisfaction percentages per edge
- Efficient scatter operations

### Message Passing
✅ Includes local satisfaction percentages in:
- c2l message updates
- l2c message normalization  
- Polarity-aware (positive/negative literals)

### Backbone Prediction
✅ Binary classification per variable:
- Label 0: negative backbone (variable must be FALSE)
- Label 1: positive backbone (variable must be TRUE)
- Label 2: free variable (not in backbone)

✅ Training objective:
- BCE loss with class weighting
- Metrics: confusion matrix, recall, precision, F1

## Data Structure

```
nsnet/
├── SATSolving/3-sat/
│   ├── train_first_4000_ArieNet/
│   │   ├── *.cnf                    # CNF files
│   │   ├── backbones.pkl            # Generated backbone labels
│   │   └── processed_bpg/           # Cached BPG graphs
│   └── valid_first_1000_ArieNet/
│       ├── *.cnf
│       ├── backbones.pkl
│       └── processed_bpg/
├── generate_backbone_labels.py      # Generate backbone labels
└── train_arienet_backbone.py        # Train ArieNet with backbone objective
```

## Model Details

- **Hidden dimension**: 128
- **Message passing rounds**: 26
- **MLP layers**: 3
- **Activation**: ReLU
- **Output**: Sigmoid probabilities for [negative, positive] backbone

## Notes

- The model automatically caches processed BPG graphs in `processed_bpg/` directories
- Backbone computation can be slow for large instances (use `--timeout` to limit)
- If a CNF file has no backbone labels, all variables default to label 2 (free)
