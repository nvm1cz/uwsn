# README experiment inputs

File input chinh:

```text
data/csv/input/experiment_inputs.csv
```

Moi dong la mot ca thuc nghiem, duoc tao tu:

```text
Input = A x B x C x D x E
```

## A. Topology mang

```text
A = distribution x space-node x seed
distribution = uniform, gaussian, exponential
space-node =
  100 m  : 20, 50, 100, 150 node
  500 m  : 100, 200, 300 node
  1000 m : 500, 1000, 1500, 2000 node
seed = 4000 -> 4014
```

So topology:

```text
3 x 11 x 15 = 495 topology
```

## B. Tham so moi truong

```text
transmission_range_m = 200
packet_size_bits = 6400
initial_energy_j = 0.5
```

So to hop:

```text
1 x 1 x 1 = 1 bo
```

## C. Thuat toan

```text
optimizer = PSO, GA, DE, LEACH, EULC, EEUMC, EBREC
```

## D. Bo tham so thuat toan

Voi PSO, GA, DE:

```text
parameter_set = baseline, energy_focused, distance_focused, balanced_low_omega
pso_particles = 20, 30
(pso_c1, pso_c2) = (1.5, 1.5)
```

So bo cho moi thuat toan PSO/GA/DE:

```text
4 x 2 x 1 = 8 bo
```

Voi LEACH, EULC, EEUMC, EBREC:

```text
parameter_set = baseline
pso_particles = 20
(pso_c1, pso_c2) = (1.5, 1.5)
```

Tong cau hinh thuat toan:

```text
PSO 8 + GA 8 + DE 8 + LEACH 1 + EULC 1 + EEUMC 1 + EBREC 1 = 28 bo
```

## E. Tham so mo phong

```text
max_rounds = 5000
stop_at_ft5 = true
min_alive_ratio = 0.95
pso_iterations = 50
cluster_head_ratio = 0.2
eulc_candidate_ratio = 0.4
```

## Tong so input theo generator hien tai

```text
495 topology x 1 bo moi truong x 28 cau hinh thuat toan = 13,860 input
```
