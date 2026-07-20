# README: Optimizer and Baseline Implementations

File nay mo ta cac thuat toan chon CH duoc cai dat ben canh model chinh PSO-EULC.

- `pso`: model chinh theo paper PSO-EULC.
- `ga`, `de`: mo rong metaheuristic de so sanh tren cung candidate set va cost.
- `leach`: baseline ngau nhien.
- `eulc`, `eeumc`, `ebrec`: baseline phan lop duoc paper dung de so sanh. `eulc` bam theo Algorithm 1 trong paper PSO-EULC. `eeumc` va `ebrec` duoc cai dat dua tren hai paper PDF dat trong thu muc `Paper`.

## Shared Pipeline

Moi thuat toan deu chay trong cung pipeline mo phong:

1. Sinh hoac doc topology node tu input co dinh.
2. Chia layer theo do sau va tinh cac dai luong moi truong: khoang cach den sink, khoang cach giua node, neighbor degree.
3. Loc tap ung vien CH bang EULC.
4. Chon tap CH bang thuat toan: `pso`, `ga`, `de`, `leach`, `eulc`, `eeumc`, hoac `ebrec`.
5. Kiem tra tinh hop le cua nghiem:
   - moi node song phai nam trong tam truyen cua it nhat mot CH;
   - moi CH phai co duong truyen len sink thong qua CH o layer nong hon hoac gui truc tiep neu trong tam;
   - nghiem sai bi phat penalty hoac bi loai.
6. Lap cum CH-member.
7. Truyen du lieu theo routing CH-to-CH ve sink.

So CH can chon duoc tinh chung cho cac thuat toan:

```text
K = max(1, round(Pc * N_active))
```

Trong do:

- `Pc` la `cluster_head_ratio`.
- `N_active` la so node con song.
- Neu so ung vien EULC nho hon `K`, `K` duoc cap theo so ung vien kha dung.

## Candidate Representation

Sau EULC, ta co tap ung vien:

```text
C = {c1, c2, ..., cM}
```

Voi PSO, GA va DE, moi nghiem deu duoc bieu dien bang vector thuc co `M` chieu:

```text
X = [x1, x2, ..., xM], xi in [0, 1]
```

Y nghia:

- moi `xi` la diem uu tien cua ung vien `ci`;
- sap xep `xi` giam dan;
- chon dung `K` ung vien co `xi` lon nhat lam CH.

Vi vay, GA va DE khong dung vector nhi phan. Chung dung cung cach ma hoa priority-vector nhu PSO de dam bao so sanh cong bang. LEACH, EULC, EEUMC va EBREC khong dung particle/chromosome/vector toi uu.

## Fitness / Cost

Tat ca optimizer su dung cung ham cost chinh theo Eq. 8 o muc candidate:

```text
C(j) = omega * (d(i,j)^2 + d(j,sink)^2) / Dmax
     + (1 - omega) * (Emax - Eres(j)) / Emax
```

Trong do:

- `j` la candidate dang duoc xet lam CH.
- `i` la candidate khac gan `j` nhat trong tap EULC candidate.
- `Dmax` la gia tri lon nhat cua distance term trong round hien tai.
- `Emax` la nang luong lon nhat trong tap EULC candidate.
- Cost cang nho thi nghiem cang tot.

Khi mot vector decode ra tap `K` CH, cost cua nghiem la:

```text
Cost_solution = mean(C(j) for j in selected_CHs) + penalty
```

Penalty duoc them ngoai cong thuc paper de dam bao nghiem co the truyen tin thuc te:

- penalty coverage: phat neu node song khong duoc CH nao phu trong transmission range;
- penalty connectivity: phat neu CH khong co duong CH-to-CH hop le len sink.

## GA Implementation

Trong GA, mot ca the duoc goi la chromosome, nhung cau truc giong priority-vector:

```text
Chromosome = [x1, x2, ..., xM], xi in [0, 1]
```

Decode:

```text
chon top-K xi lon nhat -> selected CHs
```

Quy trinh GA:

1. Khoi tao population ngau nhien trong `[0, 1]^M`.
2. Tinh fitness bang `Cost_solution`; cost nho hon la tot hon.
3. Chon cha me bang tournament selection.
4. Lai ghep uniform crossover:

```text
child_i = parentA_i hoac parentB_i theo mask ngau nhien
```

5. Dot bien Gaussian tren mot so gene:

```text
child_i = child_i + N(0, sigma)
```

6. Clip gia tri ve `[0, 1]`.
7. Giu lai elite tot nhat qua moi generation.
8. Lap den het so generation.

Tham so hien tai trong code:

- population size = `pso_particles` nhung toi thieu 2;
- generation count = `pso_iterations`;
- elite count = 2;
- crossover rate = 0.85;
- mutation rate = `min(0.25, max(1/M, 0.02))`;
- mutation sigma = 0.10;
- tournament size = 3 hoac nho hon neu population be.

GA duoc dung de so sanh voi PSO tren cung cost, cung candidate set va cung constraint penalty.

## DE Implementation

Trong DE, moi ca the la mot real-valued vector:

```text
Vector = [x1, x2, ..., xM], xi in [0, 1]
```

Decode cung theo top-K:

```text
chon top-K xi lon nhat -> selected CHs
```

Quy trinh DE:

1. Khoi tao population ngau nhien trong `[0, 1]^M`.
2. Voi moi vector muc tieu `x`, chon ba vector khac nhau `a`, `b`, `c`.
3. Tao mutant:

```text
mutant = a + F * (b - c)
```

4. Clip mutant ve `[0, 1]`.
5. Lai ghep binomial crossover giua target va mutant de tao trial vector.
6. Decode trial thanh tap CH va tinh cost.
7. Neu trial co cost nho hon hoac bang target, thay target bang trial.
8. Lap den het so generation.

Tham so hien tai trong code:

- population size = `pso_particles` nhung toi thieu 4;
- generation count = `pso_iterations`;
- differential weight `F = 0.55`;
- crossover rate `CR = 0.90`.

DE duoc dung de danh gia kha nang tim nghiem lien tuc khac PSO/GA nhung van tren cung ma hoa priority-vector.

## LEACH Implementation

LEACH trong project la baseline ngau nhien, khong dung particle/chromosome/vector toi uu.

Quy trinh LEACH:

1. Lay danh sach tat ca node con song.
2. Tinh so CH can chon:

```text
K = max(1, round(Pc * N_active))
```

3. Chon ngau nhien `K` node song lam CH.
4. Dung cung buoc finalize voi cac optimizer khac:
   - kiem tra coverage;
   - kiem tra CH connectivity;
   - lap cum CH-member;
   - routing CH-to-CH ve sink.

LEACH khong co qua trinh hoi tu lap nhieu generation. Neu can ghi convergence, code chi ghi cost cua nghiem ngau nhien ban dau tai iteration 0.

## EULC Baseline Implementation

EULC baseline bam theo Algorithm 1 trong paper PSO-EULC: tinh trong so EULC trong tung layer va chon node co trong so lon nhat lam CH cua layer, khong qua PSO.

Quy trinh:

1. Chia node song theo layer.
2. Trong moi layer, tinh nang luong trung binh `Eavg`.
3. Voi node co `Eres(i) > Eavg`, tinh:

```text
W(i) = alpha * Eres(i) / Eini
     + beta  * (1 - d(i,sink) / dmax)
     + gamma * Nk_i
```

4. Voi moi layer, chon node co `W(i)` lon nhat lam CH.
5. Lap cum va routing bang pipeline chung.

Day la baseline gan voi pha clustering cua paper: dung EULC mot minh de so voi hybrid PSO-EULC.

## EEUMC Baseline Implementation

Nguon doi chieu: `Energy-efficient unequal multi-level clustering for underwater wireless sensor networks`, Alexandria Engineering Journal, 2025, DOI `10.1016/j.aej.2024.10.026`.

Paper EEUMC tinh CH theo tung level bang nang luong chuan hoa, khoang cach chuan hoa va chi so ket hop:

```text
E_normalized = (E_node - E_min) / (E_max - E_min)
D_normalized = (D_sink - D_min) / (D_max - D_min)
M_combined = alpha * E_normalized + (1 - alpha) * D_normalized
```

Code hien tai:

- tinh `M_combined` cho tung candidate;
- chon node co `M_combined` cao trong tung layer truoc;
- bo sung candidate diem cao cho du `K`;
- lap cum theo CH gan nhat va kiem tra coverage/connectivity bang pipeline chung.

Paper EEUMC con dua routing metric:

```text
R_metric = beta * E_normalized + gamma * LQ_normalized + delta * D_normalized
```

Do simulator dang dung mot routing chung de so sanh cong bang giua cac thuat toan, metric nay duoc ghi nhan la khac biet voi paper goc, chua thay the routing chung trong cac experiment hien tai.

## EBREC Baseline Implementation

Nguon doi chieu: `Energy balanced reliable and effective clustering for underwater wireless sensor networks`, Alexandria Engineering Journal, 2023, DOI `10.1016/j.aej.2023.06.083`.

Paper EBREC dua Algorithm 1:

```text
Create network Region Cylinder with r and h
for each cylinder:
    find k = {max(R.E), min(Distance(k, C))}
    assign neighborhood nodes of k to CH k
    find a better path from source to BS in multipath
```

Code hien tai mapping vao simulator 3D nhu sau:

- moi layer duoc xem nhu mot vung/cylinder logic trong mo phong hien tai;
- `max(R.E)` duoc tinh bang residual energy chuan hoa;
- `min(Distance(k, C))` duoc xap xi bang node gan medoid/tam khoang cach cua layer nhat;
- priority score:

```text
score = 0.5 * normalized_residual_energy
      + 0.5 * layer_center_proximity
```

Sau do code chon CH theo multilevel quota va dung pipeline chung de lap cum/routing. Phan sleep scheduling va multipath trong paper khong co cong thuc du chi tiet de cai dat rieng, nen chua mo phong tach biet.

## Routing After CH Selection

Sau khi CH duoc chon, cac thuat toan deu dung cung routing:

```text
P(i,j) = epsilon * Eini(j) / Eres(j)
       + (1 - epsilon) * (d(i,j)^2 + d(j,sink)^2) / d(i,sink)^2
```

CH hien tai `i` chon next-hop `j` co `P(i,j)` nho nhat, voi dieu kien:

- `j` la CH con song;
- `j` nam trong transmission range;
- `j` o layer nong hon `i`.

Khong dung node thuong lam relay.

## Notes for Reporting

- PSO-EULC la thuat toan chinh theo paper.
- GA va DE la phan mo rong de so sanh metaheuristic tren cung candidate set va cost function.
- LEACH la baseline ngau nhien.
- EULC la baseline paper co the giai thich truc tiep tu pha EULC.
- EEUMC duoc cai theo cong thuc `M_combined` trong paper.
- EBREC duoc cai theo Algorithm 1, voi cylinder duoc mapping thanh layer/medoid trong simulator 3D hien tai.
- Tat ca thuat toan dung chung topology input, cung energy model, cung routing va cung penalty feasibility de ket qua cong bang.
