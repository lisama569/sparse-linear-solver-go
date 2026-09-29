# sparse-linear-solver-go

稀疏线性方程组 `A·x = b` 的统一迭代求解库：多种稀疏存储格式可互相转换，
多种迭代解法与预处理器共用同一入口，每次求解都返回**可解释、可复现的确定结论**。

## 目录结构

| 包 | 内容 |
|---|---|
| `sparse` | 稀疏矩阵格式（COO / CSR / CSC）、格式互转、矩阵-向量乘、残差与范数 |
| `solver` | 统一入口 `Solve`、CG / BiCGSTAB 迭代法、Jacobi / ILU(0) 预处理器、收敛状态与结果 |
| `cmd/sparse-solve-demo` | 三种格式 + 奇异系统的可运行示例 |

## 支持的矩阵格式

- **COO（Coordinate / 三元组）**：`(行索引, 列索引, 值)` 列表，允许乱序与重复坐标；
  重复坐标在转换时按**代数累加**处理，累加后恰为零的条目自动剔除。
- **CSR（Compressed Sparse Row，行压缩）**：行指针 + 列索引 + 数值，行内严格升序，适合 `A·x`。
- **CSC（Compressed Sparse Column，列压缩）**：列指针 + 行索引 + 数值，列内严格升序。

三种格式实现统一接口，可在任意方向互相转换：

```go
type Matrix interface {
    Dims() (int, int)
    NNZ() int
    MatVec(x []float64) ([]float64, error)
}
```

```go
coo, _ := sparse.NewCOO(m, n, rows, cols, vals) // 索引越界立即返回错误
csr, _ := coo.ToCSR()
csc, _ := csr.ToCSC()
y, err := sparse.MatVec(csc, x)                  // 统一的矩阵-向量乘入口
```

所有输入都会做完整性校验，三类问题以**互不相同的错误类型**报告，绝不静默当零处理：

| 错误类型 | 触发场景 |
|---|---|
| `*sparse.IndexOutOfBoundsError` | 行/列索引超出 `[0, 维度)`、负索引 |
| `*sparse.DimMismatchError` | 矩阵行列数与向量长度对不上、三元组数组长度不一致 |
| `*sparse.FormatMismatchError` | 非方阵参与求解、指针数组端点/单调性损坏、行列索引乱序或重复、未知格式/方法 |

可用标准库 `errors.As` 区分这三类原因。

## 求解方法

| 方法 | 适用场景 | 备注 |
|---|---|---|
| **CG**（共轭梯度，`MethodCG`） | 对称正定（SPD）方阵 | 检测到 `p·Ap ≤ 0` 即判定矩阵非正定并明确报失效 |
| **BiCGSTAB**（稳定双共轭梯度，`MethodBiCGSTAB`） | 一般非对称方阵 | breakdown（`rho=0`、`omega=0` 等）显式报失效 |

### 收敛判据（`Options.Criterion`）

- `RelativeResidual`：`‖b−Ax‖₂ / ‖b‖₂ ≤ Tol`
- `AbsoluteResidual`：`‖b−Ax‖₂ ≤ AbsTol`
- `BothResidual`（默认）：相对与绝对容差**同时**满足；`b=0` 时只看绝对残差

其它可配置项：`MaxIterations`（≤0 时取矩阵阶数）、`Tol`（默认 1e-8）、
`AbsTol`（默认 1e-10）、`InitialGuess`（默认零向量）、`DivergenceFactor`
（残差相对初值增长倍数阈值，默认 1e8）、`ConditionEstimate`（请求 Lanczos/Gershgorin
条件数**上界**估计，仅作病态提示，不参与判定）。

残差每步都用真实的 `b−Ax` 重新计算，避免长迭代中递推漂移导致的“假收敛”。

### 预处理器（可选，默认关闭）

| 预处理器 | 说明 |
|---|---|
| `PrecondNone` | 不预处理（`M=I`） |
| `PrecondJacobi` | `M=diag(A)`；零/极小对角元直接报奇异或明显病态 |
| `PrecondILU0` | 零填充不完全 LU，保持原稀疏结构；零主元/过小主元报奇异或明显病态 |

预处理只改变迭代路径（以 `M⁻¹` 加速收敛），**不改变原问题的解**；
测试对开启/关闭预处理的收敛解做了一致性断言。

## 终态与失败处理约定

一次求解不会“只扔回一个数组”。输入类问题以类型化 `error` 返回；
求解过程的结局一律落在 `Result.Status` 四者之一：

| `Status` | 含义 | 解向量是否可用 |
|---|---|---|
| `StatusConverged` | 已按配置判据收敛 | 是 |
| `StatusMaxIteration` | 达到 `MaxIterations` 仍未收敛 | **否**，`Converged()` 返回 false |
| `StatusDiverged` | NaN/Inf、残差爆炸、非正定（CG）、breakdown（BiCGSTAB） | 否 |
| `StatusSingular` | 整行/整列全零、预处理零主元/主元相对尺度极小（≤1e-14·max|A|） | 否 |

非收敛状态都在 `Result.FailureReason` 中给出**可区分的具体原因**
（第几步、哪个量、判定依据），`Result.Summary()` 输出单行可读结论，
方法、预处理器、判据、状态、步数、残差一应俱全，便于写进日志复现。

奇异/病态识别说明：

- 整行或整列全零、Jacobi/ILU 零主元属于**确定性**的奇异结论；
- 主元相对矩阵整体尺度小于 `1e-14` 判为“明显病态”，拒绝返回看似正常的数值；
- 仅“条件数大但非奇异”的系统（如 κ≈2000 的 SPD 矩阵）仍正常求解，
  可通过 `ConditionEstimate` 看到显著大于 1 的上界提示。

### 最小用法

```go
res, err := solver.Solve(csr, b, solver.Options{
    Method:         solver.MethodBiCGSTAB,
    Preconditioner: solver.PrecondILU0,
    Tol:            1e-10,
    MaxIterations:  500,
})
if err != nil {
    // 输入错误：errors.As 区分 IndexOutOfBounds / DimMismatch / FormatMismatch
    log.Fatal(err)
}
fmt.Println(res.Summary())
if !res.Converged() {
    log.Fatalf("求解未成功: %s: %s", res.Status, res.FailureReason)
}
// res.X 此时才是有意义的解
```

## 本地验证

```bash
# 构建与静态检查
go build ./...
go vet ./...

# 全量单测（-v 日志会打印每个用例使用的解法、预处理器与收敛判据）
go test ./... -v

# 运行示例
go run ./cmd/sparse-solve-demo
```

测试覆盖范围：

- 三种格式互转（重复坐标累加、指针布局）、CSR/CSC 结构损坏校验、SpMV 与残差；
- CG / BiCGSTAB × 无预处理 / Jacobi / ILU(0) 在 SPD 与非对称系统上的收敛；
- 同一系统分别以 COO/CSR/CSC 进入统一入口，结果一致；
- 难收敛场景（大系统 + 极小步数）→ `StatusMaxIteration`，且与发散/奇异可区分；
- 不定矩阵上 CG → `StatusDiverged`；
- 整行零、整列零、Jacobi 零主元 → `StatusSingular`；
- 高条件数但非奇异矩阵 → 正常收敛并报告条件数上界；
- 预处理开启前后解一致；三类输入错误可用 `errors.As` 区分。
