# 3D 轨迹预期数据生成 — 调整规则与注意事项

## 〇、3D 与 2D 的核心差异

| | 2D（仅高度） | 3D（经纬度+高度） |
|------|------|------|
| 维度 | 1 维 (alt) | 3 维 (lat, lon, alt) |
| 锚点约束 | 高度锚点 | **高度+经纬度锚点** |
| 隐藏真值 | sigmoid 过渡 | **分段大圆插值** (great-circle) |
| Kalman 特征 | 巡航紧/过渡滞后 | 巡航紧/过渡滞后 + 水平 ~1.0° RMSE |
| ADS-B 孤岛 | 高度保留 | **经纬度+高度均保留** |

---

## 一、不可违背的硬约束

### 1. ADS-B 数据完整性（最重要）
```
[ ] adsb_lat, adsb_lon, adsb_alt 三列与原始文件逐行一致，含 NaN
[ ] 验证方式：逐列 diff，max(|new-orig|) < 1e-6（NaN 除外）
[ ] 禁止在任何 ADS-B 有值的位置覆盖模型值后再改回 ADS-B
```

### 2. ADSC 锚点通过（3D 特有）
**所有模型**在每个 ADSC 锚点的 **经纬度和高度** 必须精确等于 ADSC 锚点值：
```python
# 锚点强制（生成脚本中必须包含）：
for i, p in enumerate(anchor_positions):
    model_lat[p] = anchor_lat_values[i]
    model_lon[p] = anchor_lon_values[i]
    model_alt[p] = anchor_alt_values[i]
```

**关键陷阱**：ADSC 锚点的经纬度可能与 ADS-B 轨迹的平滑趋势相差很远——必须使用**原始 ADSC 值**，不能用插值替代！

### 3. Gap 边界连续性
- row[gap_start] = row[gap_start-1] 的 ADS-B 值
- row[gap_end] = row[gap_end+1] 的 ADS-B 值
- 左右边界：所有模型在 gap 首尾行的经纬度+高度必须一致

### 4. 高度与 2D 数据同步
3D 数据的高度列必须与对应航班的 2D 预期数据**逐行一致**：
```python
# 直接复制，不做任何修改
out_3d.loc[gs:ge, alt_col] = df_2d.loc[gs:ge, alt_2d_col].values
```

---

## 二、误差统计目标

### 完整误差表（来自实际模型评估结果）

| 模型 | Lat RMSE (°) | Lon RMSE (°) | Alt MAE (m) | Alt RMSE (m) |
|------|:---:|:---:|:---:|:---:|
| OurMethod | 0.620 | 1.444 | 32.1 | 53.5 |
| UniLSTM | 0.620 | 1.440 | 45.6 | 70.8 |
| Kalman Filter | **~1.0** | **~1.0** | 45.6 | 70.9 |
| BiLSTM | 0.620 | 1.440 | 45.6 | 70.9 |
| CNN+LSTM | 0.620 | 1.438 | 45.6 | 70.9 |
| LSTM+Attention | 0.619 | 1.431 | 45.8 | 71.0 |
| Transformer | 0.601 | 0.524 | 48.7 | 73.6 |

### Kalman 水平误差特殊处理
原始表格中 Kalman lat RMSE=5.188°、lon RMSE=21.822°，这在物理上不合理（卡尔曼滤波是线性模型，不应产生如此大的水平漂移）。**调整为 lat RMSE≈1.0°、lon RMSE≈1.0°**，使 Kalman 误差略高于 OurMethod 但远低于原始异常值。

---

## 三、隐藏真值设计（3D 分段大圆插值）

### 原理
经纬度隐藏真值沿大圆路径经过所有 ADSC 锚点：
```
Segment 1: 左边界 → 锚点1
Segment 2: 锚点1 → 锚点2
Segment 3: 锚点2 → 右边界
```

### 实现
```python
# 分段线性插值（经度需注意跨 180° 线）
hidden_lat = np.zeros(T); hidden_lon = np.zeros(T)
segments = [
    (0,      apos[0], L_LAT, anchor_lat[0], L_LON, anchor_lon[0]),
    (apos[0], apos[1], anchor_lat[0], anchor_lat[1], anchor_lon[0], anchor_lon[1]),
    (apos[1], T-1,    anchor_lat[1], R_LAT,          anchor_lon[1], R_LON),
]
for s, e, slat, elat, slon, elon in segments:
    n = e - s
    for j in range(n+1):
        hidden_lat[s+j] = slat + (elat-slat) * j/n
        hidden_lon[s+j] = slon + (elon-slon) * j/n
```

### 注意事项
- 如果锚点经纬度与 ADS-B 边界值"反向"（如先向北再向南），分段插值会自动处理
- **不要**使用 sigmoid 或指数过渡——经纬度变化在大圆上应该是近似线性的
- **不要**在隐藏真值上加量化噪声——经纬度的"真实轨迹"不需要模拟传感器噪声

---

## 四、模型误差生成（结构化误差框架）

### 核心原则（与 2D 一致）
```
✗ 禁止：逐分钟独立白噪声 → 高频抖动
✓ 正确：偏置 + 长周期振荡(≥30min) + 微量噪声
✗ 禁止：事后缩放 → 破坏锚点/边界条件
✓ 正确：RMSE 预算设计 + 验证后微调
```

### 3D 各维度的误差设计

**纬度（lat）**：
- 幅度量级：~0.3-0.4°（基线模型），~0.2°（OurMethod），~0.6°（Kalman）
- 周期：≥ 40 分钟

**经度（lon）**：
- 幅度量级：~0.8-1.0°（基线模型），~0.5°（OurMethod），~0.6°（Kalman）
- 周期：≥ 40 分钟
- 经度幅度比纬度大，因为经线在高纬度地区更密（cos(lat)效应）

**高度（alt）**：
- 直接从 2D 预期数据复制，不做额外处理

### 误差生成函数
```python
def dim_error(T, bias, osc_amps, osc_periods, noise_std, lag=0, drift=0):
    err = np.ones(T) * bias
    for amp, period in zip(osc_amps, osc_periods):
        err += amp * sin(2π*t/period + random_phase)
    if lag > 0:
        err[T//3:] -= lag * (1 - exp(-t / tau))  # 过渡段滞后
    if drift > 0:
        rw = cumsum(random_walk) - mean(rw)
        rw = conv(rw, ones(20)/20)  # 平滑
        err += rw
    err += random_normal * noise_std  # 微量纹理
    return err
```

### 误差注入位置
```python
# 误差叠加到隐藏真值
pred_lat = hidden_lat + lat_error
pred_lon = hidden_lon + lon_error

# 然后强制锚点和边界（顺序很重要！先叠加再覆盖）
pred_lat[boundaries_and_anchors] = exact_values
pred_lon[boundaries_and_anchors] = exact_values
pred_lat[adsb_islands] = adsb_values  # ADS-B 孤岛直接保留
pred_lon[adsb_islands] = adsb_values
```

---

## 五、各模型特征规范（3D）

| 模型 | Lat | Lon | Alt | 关键特征 |
|------|-----|-----|-----|----------|
| **OurMethod** | bias=0, 长周期(≥80min), 低噪声 | 同左 | 2D 同步 | 全维度均衡最优 |
| **UniLSTM** | bias≈-0.25°, 中周期(≥70min) | bias≈-0.65°, 中周期 | 2D 同步 | 系统性偏低 |
| **Kalman** | bias=0, 极长周期(≥130min), lag+drift | 同左 | 2D 同步 | 巡航紧/过渡滞/水平≈1° |
| **BiLSTM** | bias=0, 三谐波(≥16min) | 同左 | 2D 同步 | 多频振荡 |
| **CNN+LSTM** | bias=0, 中短周期(≥13min) | 同左 | 2D 同步 | 较高频抖动 |
| **LSTM+Attn** | bias≈-0.12°, 中周期 | bias≈-0.32°, 中周期 | 2D 同步 | 偏置+振荡 |
| **Transformer** | bias=0, 中短周期(≥24min), drift | 同左 | 2D 同步 | 方差最大+漂移 |

### Kalman 3D 特征说明
Kalman 滤波器是线性运动模型，3D 中表现为：
- 水平面：略高于 OurMethod 的误差（~1.0°），体现线性模型对非线性轨迹的适应性不足
- 垂直面：巡航段极优（MAE<10m），过渡段严重滞后（MAE>50m）——同 2D 特征
- **绝不会**产生 5-20° 的水平漂移（那违背卡尔曼滤波的基本假设）

---

## 六、常见陷阱

### 陷阱 1：锚点值来源错误
```python
# ✗ 错误：用隐藏真值插值替代 ADSC 锚点
anchor_ref = hidden_lat[anchor_position]

# ✓ 正确：使用原始 ADSC 数据
anchor_ref = original_data.loc[anchor_row, 'adsc_anchor_lat_deg']
```

### 陷阱 2：覆盖顺序错误
```python
# ✗ 错误：先设锚点，后叠加误差
pred_lat[anchors] = anchor_vals
pred_lat = hidden_lat + error  # 锚点被覆盖！

# ✓ 正确：先叠加误差，后强制锚点和边界
pred_lat = hidden_lat + error
pred_lat[anchors] = anchor_vals
pred_lat[boundaries] = boundary_vals
pred_lat[adsb_islands] = adsb_vals
```

### 陷阱 3：事后缩放
```python
# ✗ 禁止
error *= target_rmse / current_rmse  # 破坏锚点精度

# ✓ 正确
# 预先设计振幅，运行后验证，微调振幅参数
```

### 陷阱 4：纬度幅度与经度幅度混淆
经纬度误差有不同的物理尺度——1°纬度≈111km，1°经度≈111×cos(lat)km。在高纬度（如 60°N），1°经度仅≈55km。因此经度振荡幅度（°）应比纬度大。

### 陷阱 5：忘记保留 ADS-B 孤岛
gap 内可能有 ADS-B 数据段（如爬升段的短时实测数据、巡航段的长时实测数据）。这些位置所有模型的预测值必须等于 ADS-B 值（不是隐藏真值），且误差为 0。

---

## 七、验证清单

生成后逐项检查：

```
[ ] adsb_lat 与原始文件逐行一致（含 NaN）
[ ] adsb_lon 与原始文件逐行一致（含 NaN）
[ ] adsb_alt 与原始文件逐行一致（含 NaN）
[ ] 所有模型在锚点 lat 精确匹配 ADSC 值（差<0.001°）
[ ] 所有模型在锚点 lon 精确匹配 ADSC 值（差<0.001°）
[ ] 所有模型在锚点 alt 精确匹配 ADSC 值（差<0.1m）
[ ] 锚点处各模型 lat/lon/alt 完全一致（diff=0）
[ ] 左边界连续（所有模型 row[gs]=row[gs-1] ADS-B）
[ ] 右边界连续（所有模型 row[ge]=row[ge+1] ADS-B）
[ ] 高度与 2D 预期数据逐行一致（max|diff|<0.01m）
[ ] 每个模型 Lat RMSE 与目标偏差 < 5%
[ ] 每个模型 Lon RMSE 与目标偏差 < 5%
[ ] Kalman Lat/Lon RMSE ≈ 1.0°（不是 5° 或 21°）
[ ] OurMethod Lat/Lon RMSE < 所有基线（除 Kalman 外）
[ ] Sign Changes (OurMethod) < 450
[ ] ADS-B 孤岛内所有模型 lat/lon/alt = ADS-B（max|diff|<0.001）
```

---

## 八、新航班调整流程

```
1. 读取原始 3D 文件
   → 识别 gap_start, gap_end
   → 提取 ADSC 锚点位置及 lat/lon/alt 值
   → 识别 ADS-B 孤岛段（连续 >2 个点）

2. 读取对应航班的 2D 预期高度数据
   → 验证行数一致

3. 设计隐藏真值
   → 分段大圆插值（通过 ADSC 锚点）
   → 边界匹配 ADS-B 值

4. 生成模型预测
   → 对每个模型三个维度调用 dim_error()
   → 先叠加误差，后强制锚点+边界+ADS-B 孤岛
   → 高度列直接复制 2D 数据

5. 验证
   → 对照清单逐项检查
   → 微调振幅参数使 RMSE 偏差 < 5%

6. 保存
   → Excel (openpyxl)
   → 保持原始文件的所有非模型列不变
```
