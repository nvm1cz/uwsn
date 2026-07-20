# Paper Clean Experiments

Protocol version: `paper_clean_v1`

## Goal

This folder contains the clean, reproducible experiment suite for the UWSN paper draft. It is separated from exploratory outputs to avoid mixing earlier trial runs with final paper evidence.

## Main Comparison

Methods: PSO-EULC, EEUMC, EBREC, EULC, and LEACH.

All methods run on the same fixed 3D topologies for each distribution, space, node count, and seed. The simulator uses common routing and energy accounting for fairness.

## Ablation

The suite also includes `pso_eulc_no_conn_penalty`, which disables only the inter-cluster connectivity penalty. This isolates the effect of the connectivity-aware extension added to the PSO-EULC objective.

## Fixed Parameters

- Spaces: `[100]`
- Distributions: `['uniform']`
- Transmission range: `200.0 m`
- Packet size: `6400 bits`
- Initial energy: `0.5 J`
- Max rounds: `1000`
- CH ratio Pc: `0.2`
- EULC candidate ratio: `0.4`
- PSO particles/iterations: `20` / `50`
- PSO inertia/c1/c2/omega: `0.7` / `1.5` / `1.5` / `0.65`

## Outputs

- `inputs/paper_clean_run_plan.csv`: exact run plan.
- `inputs/paper_clean_topologies.csv`: fixed topology metadata.
- `inputs/paper_clean_nodes.csv`: fixed node coordinates.
- `raw/paper_clean_runs.csv`: per-run summary metrics.
- `raw/paper_clean_rounds.csv`: residual energy, dead nodes, and delivered packets by round.
- `raw/paper_clean_convergence.csv`: PSO best-cost convergence events.
- `raw/paper_clean_cluster_snapshots.csv`: representative CH-member assignments.
- `figures/*.pdf` and `figures/*.png`: paper-ready figures.

## Metrics

- FT5: first round where at least 5% of nodes are dead. If FT5 is not reached by `max_rounds`, the value is treated as censored at `max_rounds` in plots.
- Residual energy: final residual energy as percentage of initial total energy.
- Dead nodes: number of dead nodes at the final simulated round.
- Runtime: wall-clock simulator runtime for one method/topology/seed run.
- Convergence: PSO best cost normalized by the initial best cost during the first clustering refresh.

## Current Status

- Completed OK runs: `6/6`
- Error runs: `0`
- OK runs by method: `{'ebrec': 1, 'eeumc': 1, 'eulc': 1, 'leach': 1, 'pso_eulc': 1, 'pso_eulc_no_conn_penalty': 1}`

## Scope and Limitations

These experiments support claims for static 3D UWSN simulation only. Node drift, acoustic channel fading, mobility, and environmental noise are not modeled. EEUMC and EBREC are implemented from their published algorithm descriptions with simulator-specific mapping for the 3D layered topology.
