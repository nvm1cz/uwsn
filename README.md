# PSO-EULC UWSN Simulator

This project implements the main PSO-EULC model for underwater wireless sensor networks, based on:

`Improving the Lifetime of UWSN Using Hybrid PSO-EULC Algorithm`

The current codebase is prepared for paper/report experiments. It keeps fixed experiment inputs, reproducible output CSV files, model documentation, and comparison baselines.

## Documentation

- `README_PSO_EULC_MODEL.md`: main PSO-EULC model workflow, equations, assumptions, and gaps versus the original paper.
- `README_OPTIMIZERS.md`: PSO, GA, DE, LEACH, EULC, EEUMC, and EBREC implementations.
- `README_EXPERIMENT_INPUTS.md`: experiment input construction.
- `EXPERIMENT_DATA_NOTES.md`: notes about experiment data and output interpretation.

## Source Modules

- `uwsn/cases.py`: simulation case metadata.
- `uwsn/run_config.py`: default tunable parameters.
- `uwsn/deployment.py`: node deployment, sink position, layer, distance, and neighbor-degree utilities.
- `uwsn/clustering.py`: EULC candidate selection.
- `uwsn/optimizers/`: PSO, GA, DE, LEACH, EULC, EEUMC, and EBREC CH-selection logic.
- `uwsn/pso.py`: compatibility wrapper for older imports.
- `uwsn/routing.py`: CH-to-CH routing and per-round transmission energy updates.
- `uwsn/simulator.py`: round-level simulation loop.
- `uwsn/metrics.py`: output metric structure.
- `uwsn/energy.py`: acoustic attenuation and energy helpers.
- `uwsn/run_batch.py` and `uwsn/cli.py`: optional batch/CLI helpers.

## Main Model

The main model is PSO-EULC:

1. Deploy or load fixed 3D node coordinates.
2. Place the sink on the water surface, default `z = 0`.
3. Compute layer, pairwise distance, distance-to-sink, and neighbor degree.
4. Use EULC to form a candidate CH set.
5. Use PSO priority vectors over the EULC candidates.
6. Decode each particle by selecting the top `K` priorities as CHs.
7. Evaluate the candidate-level paper cost plus feasibility penalties.
8. Finalize CH-member assignments.
9. Route CH aggregate packets toward shallower CHs or directly to sink.
10. Update energy and record metrics each round.

The cost and routing details are documented in `README_PSO_EULC_MODEL.md`.

## Comparison Algorithms

The project also includes GA, DE, LEACH, EULC, EEUMC, and EBREC for comparison:

- GA and DE use the same real-valued priority-vector representation as PSO.
- LEACH is a random CH-election baseline.
- EULC is the direct unequal-layer clustering baseline.
- EEUMC and EBREC are paper-comparison baselines implemented from the uploaded papers in `Paper/`; EEUMC uses the published `M_combined` metric, while EBREC maps its cylinder-based Algorithm 1 to the simulator's layer/medoid representation.
- All algorithms share the same topology, EULC candidate setup where applicable, feasibility checks, routing, energy model, and output metrics.

See `README_OPTIMIZERS.md`.

## Fixed Experiment Inputs

Primary input files:

- `data/csv/input/experiment_inputs.csv`
- `data/csv/input/experiment_topologies.csv`
- `data/csv/input/experiment_nodes.csv`
- `data/csv/input/experiment_manifest.csv`

These files are intended to keep topology generation separate from model execution, so experiments can be reproduced without regenerating random node coordinates.

## Main Outputs

Primary output files:

- `outputs/data/csv/output/experiment_outputs.csv`
- `outputs/data/csv/output/experiment_convergence.csv`
- `outputs/data/csv/output/experiment_cluster_heads.csv`
- `outputs/data/csv/output/experiment_cluster_members.csv`

Primary figure folders:

- `outputs/figures/`
- `outputs/report/`

Historical logs, backups, and archives are not part of the main paper result set.

## Run Experiments

Run from the fixed input CSV files:

```powershell
python run_experiment_inputs.py
```

Run in parallel:

```powershell
python run_experiment_inputs_parallel.py
```

Generate fixed experiment inputs:

```powershell
python build_experiment_inputs.py
```

For small smoke tests, edit `uwsn/run_config.py` and instantiate `PsoEulcSimulator` directly.

## Important Implementation Notes

- `cluster_head_ratio` implements the paper's `Pc` and is used to compute `K = round(Pc * N_active)`.
- `eulc_candidate_ratio` is an implementation control used to limit the EULC candidate pool before PSO.
- Coverage and CH-connectivity penalties are controlled additions that prevent infeasible clustering/routing solutions.
- The implementation should be described as a controlled reimplementation and extension of the paper, not as an exact 100% reproduction.
