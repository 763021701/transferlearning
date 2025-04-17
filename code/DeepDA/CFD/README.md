# Characteristic Frequency Discrepancy (CFD)

## Overview
CFD is a domain adaptation method that measures and minimizes the discrepancy between source and target domains in the frequency domain. It analyzes both amplitude and phase differences of feature distributions when projected to the frequency domain, providing a comprehensive measurement of domain discrepancy.

## Method
The CFD loss consists of two components:
1. **Amplitude Discrepancy**: Measures the differences in the magnitude of characteristic frequencies between domains
2. **Phase Discrepancy**: Measures the differences in the phase of characteristic frequencies between domains

The method uses a learnable sampling network to select discriminative frequency components, which adaptively finds frequencies where the source and target domains differ the most.

## Parameters
- `cfd_alpha`: Weight for amplitude discrepancy term (0-1)
- `cfd_beta`: Weight for phase discrepancy term (0-1)
- `t_batchsize`: Batch size for the sampling network that generates frequency components

## Usage
To use CFD loss for domain adaptation, set the following parameters in your config file:

```yaml
transfer_loss: cfd
cfd_alpha: 0.5
cfd_beta: 0.5
t_batchsize: 2048
```

## Running an Experiment
```bash
python main.py --config CFD/config.yaml
```
