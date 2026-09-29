# sparse-linear-solver-go

在统一入口下求解稀疏线性方程组 `A*x = b`：支持多种稀疏存储格式、多种迭代解法、
可选预处理器，并对每次求解给出可解释、可复现的最终状态。

## 支持的矩阵格式

包 `sparse` 中所有格式都实现统一接口 `Matrix`（`Dims` / `NNZ` / `MatVec`）：

| 格式 | 类型 | 说明 |
|------|------|------|
| COO  | `sparse.COO`  | 坐标格式，构造入口；同一坐标的重复项在压缩时累加 |
| CSR  | `sparse.CSR`  | 压缩行存储，行访问与 MatVec 高效 |
| CSC  | `sparse.CSC`  | 压缩列存储 |

三种格式可互相转换：`ToCOO` / `ToCSR` / `ToCSC`，转换保持矩阵语义不变，
同一逻辑矩阵在任何格式下 `MatVec` 结果一致。

### 输入校验约定

非法输入不会被默默当零处理，错误类别可区分（`errors.Is`）：

- `ErrDimensionMismatch`：矩阵非方阵、`len(x)`/`len(b)` 与矩阵阶数不符；
- `ErrIndexOutOfRange`：COO 条目行列索引越界（含负数）；
- `ErrZeroPivot`：迭代法或预处理器遇到零/NaN/Inf 对角元。

## 求解方法

统一入口：

```go
r := sparse.Solve(a, b, sparse.CG{}, sparse.Config{
    Tolerance:      1e-10,                       // 相对残差判据 |r|/|b| <= tol
    MaxIterations:  10000,                       // 步数上限
    Preconditioner: &sparse.JacobiPreconditioner{}, // 可选，nil 表示不用
})
```

| 方法 | 类型 | 适用 |
|------|------|------|
| Jacobi        | `sparse.Jacobi`       | 对角非零，对角占优时收敛 |
| Gauss-Seidel  | `sparse.GaussSeidel`  | 对角非零，内部转 CSR 行访问 |
| CG / PCG      | `sparse.CG`           | 对称正定；可配预处理器 |

`Tolerance <= 0` 或 `MaxIterations <= 0` 时使用缺省值（`1e-10` / `10000`）。

### 收敛与失败约定

每次求解返回 `sparse.Result`，`Status` 三选一，互斥：

| 状态 | 含义 |
|------|------|
| `Converged`             | 步数内满足 `|r|/|b| <= Tolerance`，`X` 有效 |
| `MaxIterationsReached`  | 达到步数上限仍未收敛，`Reason` 给出最终残差 |
| `Failed`                | 数值发散或解算失效，`Reason` 给出具体原因 |

`Failed` 覆盖：输入维度不符、右端含 NaN/Inf、零主元（奇异）、
CG 的 `p^T A p <= 0`（非 SPD）、迭代中出现 NaN/Inf、
相对残差放大超过 `1e6` 倍（判定发散）、预处理器构造失败。
任何非 `Converged` 结果都带有可读的 `Reason`，不会返回"看着正常但无意义"的数值。

### 预处理

`JacobiPreconditioner` 用 `A` 的对角线构造 `M`，在 CG 中即 PCG。
预处理只改变迭代路径、不改变原问题的解；矩阵对角含零/NaN/Inf 时
`Setup` 返回 `ErrZeroPivot`，求解以 `Failed` 终止。

## 本地验证

```sh
go build ./...
go vet ./...
go test ./sparse/ -v
```

测试覆盖：格式互转与 MatVec 一致性、维度/索引错误分类、三种解法在三种格式下
收敛到同一真解、预处理加速且不改解、难收敛（步数耗尽）、奇异矩阵（零主元）、
病态矩阵（Hilbert）、非对称/不定矩阵被 CG 拒绝、非法右端。
单测日志会打印每个用例使用的解法、收敛判据（tol / maxIter / 预处理器）
以及最终状态与失败原因。
