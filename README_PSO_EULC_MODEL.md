# README: Main PSO-EULC Model

File nay mo ta chi tiet model chinh `PSO-EULC` dang duoc cai dat trong project. GA, DE, LEACH, EULC, EEUMC va EBREC la cac thuat toan so sanh, duoc mo ta rieng trong `README_OPTIMIZERS.md`.

## 1. Muc Tieu Model

Model mo phong mang cam bien duoi nuoc 3D, trong do:

- node cam bien nam trong khoi nuoc 3D;
- sink nam tren mat nuoc, mac dinh tai `(width/2, height/2, 0)`;
- mat nuoc co do sau `z = 0`;
- node co do sau lon hon thi nam sau hon;
- he thong chay theo round;
- sau moi `recluster_interval` round, model chon lai CH va cum;
- muc tieu la keo dai lifetime, tang FT5/FND, giam tieu hao nang luong va dam bao du lieu co duong truyen ve sink.

## 2. Input Chinh

Moi lan chay can cac thong tin:

- kich thuoc khong gian: `width_m`, `height_m`, `depth_m`;
- so node: `node_count`;
- vi tri node 3D;
- vi tri sink;
- nang luong ban dau: `initial_energy`;
- packet size;
- transmission range;
- cac trong so EULC: `alpha`, `beta`, `gamma`;
- cac tham so PSO: particle count, iteration, inertia `w`, `c1`, `c2`, `omega`;
- ti le CH: `cluster_head_ratio`;
- ti le candidate EULC: `eulc_candidate_ratio`;
- seed/topology neu can tai lap thuc nghiem.

Neu khong doc tu file input co dinh, node duoc sinh theo mot trong ba phan phoi:

- `uniform`;
- `gaussian`;
- `exponential`.

## 3. Khoi Tao Mang

Tai dau moi lan simulation:

1. Sinh hoac doc toa do node.
2. Khoi tao nang luong moi node:

```text
E(node) = E_initial
```

3. Tinh layer theo do sau `z`.
4. Tinh khoang cach moi node den sink:

```text
d(i, sink)
```

5. Tinh ma tran khoang cach giua moi cap node:

```text
d(i, j)
```

6. Tinh neighbor degree cua moi node dua tren so node nam trong transmission range.

## 4. Chia Layer Theo Do Sau

Code dung do sau `z` cua node:

```text
surface = 0
depth = z
```

Layer duoc tinh theo:

```text
neu depth < R0:
    layer = 1
nguoc lai:
    layer = 2 + floor((depth - R0) / (R0 + layer_spacing))
```

Y nghia:

- layer nho hon la nong hon, gan mat nuoc hon;
- layer lon hon la sau hon;
- routing CH-to-CH chi di tu layer sau len layer nong hon.

## 5. EULC Candidate Selection

EULC khong chon CH truc tiep, ma loc ra tap ung vien CH truoc khi dua vao PSO.

Trong moi layer:

1. Lay cac node con song trong layer.
2. Tinh nang luong trung binh cua layer:

```text
Eavg(layer)
```

3. Chi giu node co:

```text
Eres(i) > Eavg(layer)
```

4. Tinh trong so EULC cho node ung vien:

```text
W(i) = alpha * Eres(i) / Eini
     + beta  * (1 - d(i,sink) / dmax)
     + gamma * neighbor_degree(i)
```

Trong do:

- `alpha`: uu tien node nhieu nang luong;
- `beta`: uu tien node gan sink hon;
- `gamma`: uu tien node co mat do/degree tot hon;
- `neighbor_degree` da duoc chuan hoa.

5. Trong moi layer, sap xep node theo `W(i)` giam dan.
6. Lay top theo `eulc_candidate_ratio`.

Neu layer khong co node nao thoa `Eres > Eavg`, code fallback bang cach lay mot so node nang luong cao nhat trong layer.

## 6. Unequal Clustering Candidate Filtering

Sau khi co danh sach candidate, code ap dung co che loc gan nhau theo ban kinh canh tranh:

```text
R_i = R0 * (1 - c * (d(i,sink) - dmin) / (dmax - dmin))
```

Trong do:

- `R0 = layer_r0_m`;
- `c = 0.5`;
- node gan sink co ban kinh canh tranh khac node xa sink;
- muc tieu la tranh cac CH ung vien qua gan nhau.

Code sap xep candidate theo EULC weight. Voi moi candidate:

- neu candidate khong nam trong ban kinh canh tranh cua candidate da chon, giu lai;
- neu qua gan candidate da chon, bo qua.

Sau do code bo sung coverage candidate neu can: neu co node song khong duoc candidate nao phu trong transmission range, node do duoc them vao candidate set. Phan nay la bo sung thuc nghiem de tranh mat node ngay tu dau.

## 7. Particle Representation Trong PSO

Sau EULC, ta co tap candidate:

```text
C = {c1, c2, ..., cM}
```

Moi particle `j` trong PSO gom:

```text
X_j = [x_j1, x_j2, ..., x_jM], x_ji in [0, 1]
V_j = [v_j1, v_j2, ..., v_jM], v_ji in [-Vmax, Vmax]
```

Y nghia:

- moi chieu ung voi mot candidate EULC;
- `x_ji` la diem uu tien cua candidate `ci`;
- diem cang lon thi candidate cang co kha nang duoc decode thanh CH;
- `V_j` dieu chinh vector uu tien o iteration tiep theo.

Trong code:

```text
Vmax = 1.0
position range = [0, 1]
```

## 8. So CH Can Chon

So CH muc tieu duoc tinh dong theo so node con song:

```text
K = max(1, round(Pc * N_active))
```

Trong do:

- `Pc = cluster_head_ratio`;
- `N_active` la so node con song;
- neu so candidate `M < K`, thi `K` duoc cap bang `M`.

## 9. Decode Particle Thanh Tap CH

Voi particle:

```text
X = [x1, x2, ..., xM]
```

Decode nhu sau:

1. Sap xep cac gia tri `xi` giam dan.
2. Lay dung `K` candidate co `xi` lon nhat.
3. Cac candidate nay tro thanh selected CH.

Vi du:

```text
C = [node1, node2, node3, node4]
X = [0.20, 0.85, 0.40, 0.90]
K = 2
```

Sap xep giam dan:

```text
node4 = 0.90
node2 = 0.85
node3 = 0.40
node1 = 0.20
```

Ket qua:

```text
CH = {node4, node2}
```

## 10. Ham Cost Chinh Theo Paper

Cost chinh dang dung la Eq. 8 o muc candidate.

Voi moi candidate `j` duoc chon lam CH:

```text
C(j) = omega * (d(i,j)^2 + d(j,sink)^2) / Dmax
     + (1 - omega) * (Emax - Eres(j)) / Emax
```

Trong do:

- `j` la candidate dang duoc danh gia lam CH;
- `i` la candidate khac gan `j` nhat trong tap EULC candidate;
- `d(i,j)` la khoang cach giua hai candidate;
- `d(j,sink)` la khoang cach tu candidate `j` den sink;
- `Dmax` la gia tri lon nhat cua distance term trong round hien tai;
- `Emax` la nang luong lon nhat trong tap EULC candidate;
- `omega` la trong so can bang khoang cach va nang luong.

Distance term:

```text
DI_j = (d(i,j)^2 + d(j,sink)^2) / Dmax
```

Energy term:

```text
EI_j = (Emax - Eres(j)) / Emax
```

Cost cua mot candidate:

```text
C(j) = omega * DI_j + (1 - omega) * EI_j
```

Cost cua mot particle sau khi decode ra `K` CH:

```text
Cost_particle = mean(C(j) for j in selected_CHs)
```

Cost cang nho thi nghiem cang tot.

## 11. Penalty Them Ngoai Paper

Ngoai cost goc, code them penalty de loai nghiem khong kha thi trong thuc te.

### 11.1. Coverage Penalty

Sau khi decode CH, moi node con song phai co it nhat mot CH trong transmission range.

Neu co node khong duoc phu:

```text
penalty += very_large_value
```

Muc tieu:

- khong co node bi bo ngoai cum;
- khong co member nam xa CH hon transmission range;
- tranh nghiem co cost dep nhung khong truyen duoc du lieu.

### 11.2. CH Connectivity Penalty

Moi CH phai:

- gui truc tiep duoc ve sink neu sink trong transmission range; hoac
- tim duoc CH khac o layer nong hon trong transmission range.

Neu CH khong co duong di len sink:

```text
penalty += connectivity_penalty
```

Phan penalty nay la bo sung thuc nghiem, khong phai cong thuc goc trong paper. No giup dam bao routing hop ly khi visualize 3D va khi tinh packet delivery.

## 12. PSO Optimization Loop

Moi particle co `position` va `velocity`.

Khoi tao:

```text
X_j random trong [0, 1]^M
V_j = 0
```

Tai moi iteration:

1. Decode moi particle thanh tap CH.
2. Tinh cost cua tap CH.
3. Cap nhat `pbest` neu particle tot hon lich su cua no.
4. Cap nhat `gbest` neu particle tot hon nghiem tot nhat toan swarm.
5. Cap nhat velocity:

```text
v = w * v
  + c1 * r1 * (pbest - x)
  + c2 * r2 * (gbest - x)
```

6. Clip velocity ve:

```text
[-Vmax, Vmax]
```

7. Cap nhat position:

```text
x = x + v
```

8. Clip position ve:

```text
[0, 1]
```

9. Lap den het `pso_iterations`.

Mac dinh:

```text
particles = 20
iterations = 50
w = 0.7
c1 = 1.5
c2 = 1.5
omega = 0.65
```

## 13. Stagnation Restart

Code co them co che restart khi swarm bi dung qua lau.

Neu sau mot so iteration khong cai thien `gbest`, code:

- tim mot phan cac particle te nhat;
- khoi tao lai position cua chung ngau nhien;
- reset velocity;
- tinh lai cost.

Muc tieu:

- tranh PSO ket som o nghiem cuc bo;
- tang co hoi tim duoc nghiem hop le khi co penalty coverage/connectivity.

## 14. Finalize Tap CH

Sau khi PSO ket thuc:

1. Decode `gbest` thanh tap CH.
2. Kiem tra coverage.
3. Kiem tra connectivity.
4. Neu hop le, lap cum CH-member.
5. Neu khong hop le, bao loi thay vi am tham chap nhan nghiem sai.

Lap cum:

- moi node song duoc gan vao CH gan nhat trong transmission range;
- CH cung co the la member cua chinh no;
- ket qua luu dang:

```text
CH -> list(member nodes)
```

## 15. Reclustering Theo Round

He thong chay tung round.

Code se recluster khi:

- round dau tien;
- den chu ky `recluster_interval`;
- CH hien tai da chet;
- chua co assignment hop le.

Mac dinh:

```text
recluster_interval = 20
```

Nghia la clustering va routing path co the thay doi sau moi chu ky recluster.

## 16. Intra-Cluster Transmission

Trong moi round:

1. Moi node song ton nang luong broadcast control packet.
2. Member gui data packet den CH cua minh.
3. CH ton nang luong receive packet tu member.
4. CH aggregate du lieu.

Nang luong gui phu thuoc vao:

- packet size;
- khoang cach;
- attenuation acoustic;
- contention multiplier neu bat contention.

## 17. Inter-Cluster Routing

Sau khi CH aggregate du lieu, CH gui packet ve sink.

Neu CH trong transmission range cua sink:

```text
CH -> sink
```

Neu khong, CH chon next-hop la CH khac o layer nong hon.

Routing cost theo Algorithm 1:

```text
P(i,j) = epsilon * Eini(j) / Eres(j)
       + (1 - epsilon) * (d(i,j)^2 + d(j,sink)^2) / d(i,sink)^2
```

Trong do:

- `i` la CH hien tai;
- `j` la CH ung vien lam next-hop;
- `j` phai con song;
- `j` phai nam trong transmission range cua `i`;
- `j` phai o layer nong hon `i`;
- chon `j` co `P(i,j)` nho nhat.

Code lap lai quy tac nay cho den khi:

- gap CH co the gui truc tiep ve sink; hoac
- khong tim duoc route hop le.

Khong dung node thuong lam relay.

## 18. Energy Model Trong Round

Trong round, nang luong bi tru theo cac buoc:

1. Broadcast control packet.
2. Member transmit data den CH.
3. CH receive data tu member.
4. CH aggregate data.
5. CH transmit aggregate packet den next-hop hoac sink.
6. Relay CH receive packet neu co.

Neu nang luong node <= `dead_energy_threshold_j`, node duoc xem la chet va khong tiep tuc truyen.

## 19. Metrics Dau Ra

Model ghi cac metric chinh:

- residual energy theo round;
- dead nodes theo round;
- packets received theo round;
- FND: first node death;
- HND: half nodes death;
- LND: last node death;
- FT5: round khi 5% node chet;
- pso convergence neu bat tracking;
- cluster snapshots gom CH va members.

## 20. Diem Giu Dung Paper Va Diem Bo Sung

Phan bam paper:

- chia layer theo do sau;
- EULC candidate selection dua tren residual energy, distance to sink, neighbor degree;
- PSO tim CH tren tap candidate;
- cost Eq. 8 voi distance-energy tradeoff;
- routing CH-to-CH bang `P(i,j)`;
- cost cang nho cang tot.

Phan bo sung de model thuc te hon:

- decode top-K ro rang tu priority vector;
- penalty coverage;
- penalty CH connectivity;
- candidate coverage augmentation de tranh node bi bo sot;
- stagnation restart trong PSO;
- luu input/output va snapshot de tai lap thuc nghiem;
- ho tro visualization 3D.

## 21. Gap So Voi Paper Goc

Phan nay ghi ro cac diem model hien tai khac hoac bo sung so voi paper goc `Improving the Lifetime of UWSN Using Hybrid PSO-EULC Algorithm`.

### 21.1. Gap Do Paper Mo Ta Chua Du Chi Tiet

Paper khong mo ta that ro mot so buoc trien khai, nen code phai chon mot cach cu the de chay duoc simulation.

1. Particle decoding

Paper noi PSO toi uu vi tri/node selection, nhung khong noi chi tiet cach anh xa vector PSO sang tap nhieu CH. Code hien tai dung priority-vector:

```text
X = [x1, x2, ..., xM], xi in [0, 1]
chon top-K xi lon nhat lam CH
```

Day la cach trien khai ro rang va tai lap duoc, nhung la phan cu the hoa them so voi mo ta trong paper.

2. So CH can chon

Paper co `cluster-head ratio Pc` trong phan Require cua EULC, nhung cac dong thuat toan khong viet ro cong thuc dung `Pc` de tinh so CH moi round. Code hien tai cu the hoa `Pc` bang:

```text
K = max(1, round(Pc * N_active))
```

Day la cach trien khai co kiem soat, giup so CH thay doi theo so node con song.

3. Cost cua particle khi co nhieu CH

Paper dua cost o dang candidate-level `C(j)`, nhung khong noi ro cach tong hop cost khi particle decode ra nhieu CH. Code hien tai dung:

```text
Cost_particle = mean(C(j) for j in selected_CHs)
```

Day la cach tong hop nhat quan voi muc tieu minimize cost.

4. Xu ly nghiem khong kha thi

Paper khong noi ro neu tap CH duoc chon khong phu duoc node hoac khong co duong routing thi xu ly the nao. Code hien tai them penalty va reject nghiem khong hop le.

### 21.2. Phan Bo Sung De Dam Bao Tinh Thuc Te

1. Coverage penalty

Bo sung de tranh truong hop node song khong nam trong cum nao hoac member xa CH hon transmission range.

2. CH connectivity penalty

Bo sung de tranh truong hop CH khong co CH nong hon trong tam truyen va cung khong gui truc tiep duoc ve sink.

3. Candidate coverage augmentation

Sau EULC, neu tap candidate khong the phu het node song, code bo sung node can thiet vao candidate set. Muc tieu la tranh loi do EULC loc qua chat lam mat kha nang lap cum.

4. Stagnation restart trong PSO

Bo sung de tranh PSO ket som tai nghiem cuc bo, nhat la khi co penalty feasibility lon.

5. Input/output co dinh de tai lap thuc nghiem

Paper chi bao cao ket qua; code hien tai luu input topology, output metric, convergence va cluster snapshot de co the tai lap va giai thich ket qua.

6. Visualization 3D

Bo sung de quan sat truc quan node, CH, member-to-CH va CH routing theo round. Day la cong cu minh hoa, khong phai thanh phan thuat toan goc.

### 21.3. Phan Chua Co Trong Model Hien Tai

Mot so yeu to UWSN thuc te chua duoc mo phong day du:

1. Node mobility / drift

Node hien tai gan nhu co dinh theo topology trong mot simulation. Chua mo phong dong chay lam node troi dat theo thoi gian.

2. Link quality ngau nhien

Routing hien dua tren khoang cach, layer va nang luong. Chua co mo hinh packet loss ngau nhien theo kenh am hoc.

3. Collision/MAC chi moi la xap xi

Code co contention multiplier, nhung chua mo phong MAC layer chi tiet.

4. Energy-safe precheck trong tung round chua hoan hao

Topology duoc kiem tra hop le truoc khi truyen. Tuy nhien trong mot round, neu CH/relay het nang luong sau khi nhan/truyen mot phan du lieu, packet co the bi mat. Day la diem can ghi ro neu bao cao ve reliability muc packet.

5. Ket qua khong phai reproduction 100% paper

Do paper thieu chi tiet ve topology seed, cach decode particle, cach xu ly nghiem loi va mot so tham so, ket qua hien tai nen duoc trinh bay la reimplementation/experimental extension, khong nen noi la reproduce y het paper.

### 21.4. Cach Nen Trinh Bay Khi Bao Ve

Nen noi:

```text
Em bam theo core PSO-EULC cua paper: EULC candidate selection, PSO optimization, cost distance-energy va routing P(i,j).
Tuy nhien paper khong mo ta chi tiet mot so buoc trien khai nhu decode particle, so CH moi round va xu ly nghiem khong kha thi.
Vi vay em cu the hoa cac buoc nay va them penalty coverage/connectivity de dam bao ket qua truyen tin hop ly trong mo phong.
```

Khong nen noi:

```text
Code reproduce chinh xac 100% paper.
```

Nen noi:

```text
Day la ban cai dat lai va mo rong co kiem soat dua tren paper goc.
```

## 22. Tom Tat Mot Round

Mot round co the tom tat nhu sau:

```text
if need_recluster:
    candidates = EULC_select_candidates()
    particles = initialize_priority_vectors()
    for iteration in PSO_iterations:
        selected_CHs = decode_top_K(particle)
        cost = paper_cost(selected_CHs) + penalty
        update_pbest_gbest()
        update_velocity_position()
    CHs, members = finalize(gbest)

member_nodes -> CH
CH aggregate
CH -> shallower CH -> ... -> sink
update energy
record metrics
```

Day la ban chinh thuc cua model PSO-EULC trong project hien tai.
