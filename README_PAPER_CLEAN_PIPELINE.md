# Paper-Clean Experiment Pipeline

This is the official experiment path for the UWSN paper draft. It is separated from older exploratory outputs.

## One-Sentence Contribution

We evaluate a connectivity-aware PSO-EULC adaptation for static 3D UWSNs, where PSO selects cluster heads from EULC candidates and penalizes disconnected CH chains so that aggregated packets can be forwarded through shallower CHs toward the surface sink.

## Official Experiment Suite

Components:

- Main comparison: `PSO-EULC`, `EEUMC`, `EBREC`, `EULC`, `LEACH`.
- Ablation: `PSO-EULC w/o connectivity penalty`.
- Spaces: `100 m`, `500 m`.
- Node sets: `20,50,100,150` for `100 m`; `100,200,300` for `500 m`.
- Distributions: `uniform`, `gaussian`, `exponential`.
- Seeds: default `10` seeds per topology setting.
- Rounds: default `1000`.
- Recluster interval: `20` rounds.
- Transmission range: `200 m`.
- Packet size: `6400 bits`.
- Initial energy: `0.5 J`.

Expected full run count:

`2 space groups x node sets x 3 distributions x 10 seeds x 6 methods = 1260 method runs`

## Run Commands

Full official run:

```powershell
cd D:\NVM_20235783\20252\GR1\UWSN\src
python run_paper_clean_experiments.py --output-dir outputs/paper_clean --runs 10 --max-rounds 1000
```

Resume the same run after interruption:

```powershell
cd D:\NVM_20235783\20252\GR1\UWSN\src
python run_paper_clean_experiments.py --output-dir outputs/paper_clean --runs 10 --max-rounds 1000
```

The runner skips completed OK `run_id`s unless `--no-resume` is passed.

Smoke test:

```powershell
cd D:\NVM_20235783\20252\GR1\UWSN\src
python run_paper_clean_experiments.py --output-dir outputs/paper_clean_smoke --runs 1 --spaces 100 --distributions uniform --max-rounds 50 --max-cases 6
```

Plot only from existing results:

```powershell
cd D:\NVM_20235783\20252\GR1\UWSN\src
python run_paper_clean_experiments.py --output-dir outputs/paper_clean --plot-only
```

## Outputs

- `outputs/paper_clean/inputs/paper_clean_run_plan.csv`
- `outputs/paper_clean/inputs/paper_clean_topologies.csv`
- `outputs/paper_clean/inputs/paper_clean_nodes.csv`
- `outputs/paper_clean/raw/paper_clean_runs.csv`
- `outputs/paper_clean/raw/paper_clean_rounds.csv`
- `outputs/paper_clean/raw/paper_clean_convergence.csv`
- `outputs/paper_clean/raw/paper_clean_cluster_snapshots.csv`
- `outputs/paper_clean/raw/paper_clean_summary.csv`
- `outputs/paper_clean/figures/*.pdf`
- `outputs/paper_clean/figures/*.png`
- `outputs/paper_clean/README.md`

## Figures Generated

- `fig01_ft5_by_node_count`: network lifetime by FT5.
- `fig02_residual_energy_by_node_count`: final residual energy.
- `fig03_dead_nodes_by_node_count`: final dead nodes.
- `fig04_runtime_by_node_count`: wall-clock runtime.
- `fig05_pso_convergence_first_refresh`: PSO convergence in the first clustering refresh.
- `fig06_connectivity_penalty_ablation_ft5`: ablation of the connectivity penalty.

## Paper-Safe Claim Scope

Use claims only within this scope:

- Static 3D UWSN simulation.
- Fixed surface sink at depth `0`.
- No node mobility/drift.
- No acoustic fading/noise model beyond the implemented distance-based energy model.
- EEUMC and EBREC are simulator-specific implementations based on their published algorithm descriptions.

Do not claim real-sea deployment validity unless mobility and acoustic channel effects are added later.
