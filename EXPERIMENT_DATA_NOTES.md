# Experiment Data Notes

## Input Files

- Main input: `data/csv/input/experiment_inputs.csv`
- Fixed topologies: `data/csv/input/experiment_topologies.csv`
- Fixed node coordinates: `data/csv/input/experiment_nodes.csv`
- Manifest: `data/csv/input/experiment_manifest.csv`

Each row in `experiment_inputs.csv` is one reproducible experiment case. It combines topology, optimizer, parameter set, space size, node count, distribution, seed, transmission range, packet size, and initial energy.

With the current generator, a full input rebuild creates:

```text
495 fixed topologies x 28 algorithm configurations = 13,860 inputs
```

## Output Files

- Summary output: `outputs/data/csv/output/experiment_outputs.csv`
- Convergence output: `outputs/data/csv/output/experiment_convergence.csv`
- Representative CH snapshots: `outputs/data/csv/output/experiment_cluster_heads.csv`
- Representative cluster-member snapshots: `outputs/data/csv/output/experiment_cluster_members.csv`

`experiment_outputs.csv` stores FT5, FND, HND, LND, dead/alive nodes at stop, residual energy, packets received, recluster count, routing edge count, and runtime.

Detailed routing edges are no longer saved by default to reduce runtime and output size.

## Algorithms

Main model:

```text
PSO-EULC
```

Comparison algorithms:

```text
GA, DE, LEACH, EULC, EEUMC, EBREC
```

GA and DE are metaheuristic extensions using the same priority-vector representation and cost as PSO. LEACH is a random baseline. EULC is the direct clustering baseline. EEUMC and EBREC are implemented from the uploaded papers in `Paper/`; EEUMC follows the published `M_combined` metric, while EBREC follows Algorithm 1 with the cylinder center represented by the current simulator layer medoid.

## Report Figures

Core figures for evaluation:

- FT5 comparison: compares network lifetime; higher FT5 is better.
- FT5 standard deviation: compares stability of FT5 across seeds/topologies; lower is more stable.
- Runtime comparison: compares execution cost; lower runtime is faster.
- Convergence comparison: compares iterative optimizers PSO, GA, and DE by best cost over iteration; lower relative cost is better.

LEACH, EULC, EEUMC, and EBREC do not have true iterative convergence curves because they are direct baselines.
