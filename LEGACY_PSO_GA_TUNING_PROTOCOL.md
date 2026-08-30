# Legacy Static PSO/GA Tuning Scope

This branch is reserved for the historical static UWSN simulator:

- no mobility;
- no ambient-noise/SNR/BER/PER model;
- no retransmission;
- no delay model or energy-delay objective;
- shared EULC candidate selection, distance-energy cost and greedy routing;
- PSO and GA are compared with the same topology, candidate set, cost and seed.

## Confirmed scenario matrix

The following two historical cases are explicitly excluded:

- `100 x 100 x 100 m, 20 nodes`;
- `1000 x 1000 x 1000 m, 2000 nodes`.

The retained space-node combinations are:

| Space (m) | Node counts |
|---|---|
| `100 x 100 x 100` | `50, 100, 150` |
| `500 x 500 x 500` | `100, 200, 300` |
| `1000 x 1000 x 1000` | `500, 1000, 1500` |

Each space-node combination has four packet-energy cases:

- `6400 bits, 1.0 J/node`;
- `6400 bits, 0.5 J/node`;
- `4000 bits, 1.0 J/node`;
- `4000 bits, 0.5 J/node`.

Therefore:

- 9 space-node combinations;
- 36 scenarios per optimizer;
- 36 PSO scenarios and 36 GA scenarios;
- 72 optimizer-scenario combinations before repeated seeds/hyperparameter trials.

This matrix is the required source of truth for future legacy PSO/GA tuning.
