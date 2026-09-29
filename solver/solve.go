package solver

import (
	"errors"
	"fmt"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// Solve 是稀疏线性方程组 A*x = b 的统一求解入口。
//
// A 可以是 sparse 包中的任意存储格式（COO/CSR/CSC），内部统一转换为 CSR。
// 输入层面的格式不匹配、维度对不上、索引越界以 error 返回（可通过
// errors.As 区分 sparse.FormatMismatchError / DimMismatchError /
// IndexOutOfBoundsError）；求解层面的终态一律落在 Result.Status：
// 收敛、达到步数上限、数值发散/失效、奇异/病态。
func Solve(A sparse.Matrix, b []float64, opts Options) (*Result, error) {
	if sparse.IsNilMatrix(A) {
		return nil, &sparse.FormatMismatchError{
			Operation: "Solve", Got: "<nil>", Expect: "非空稀疏矩阵",
			Detail: "系数矩阵不能为空",
		}
	}
	rows, cols := A.Dims()

	// 1. 输入校验与默认值。
	opts, err := withDefaults(A, b, opts)
	if err != nil {
		return nil, err
	}

	// 2. 统一转换为 CSR（任何格式损坏都会在这里以显式错误暴露）。
	csr, err := toCSR(A)
	if err != nil {
		return nil, err
	}

	// 3. 奇异结构的确定性识别：整行/整列全零必然奇异。
	if err := detectStructuralSingularity(csr); err != nil {
		res := &Result{
			Status:         StatusSingular,
			Method:         opts.Method,
			Criterion:      opts.Criterion.String(),
			Preconditioner: opts.Preconditioner,
			X:              make([]float64, rows),
			MaxIterations:  opts.MaxIterations,
			Tol:            opts.Tol,
			AbsTol:         opts.AbsTol,
			FailureReason:  err.Error(),
		}
		return res, nil
	}

	// 4. 初值。
	x0 := make([]float64, rows)
	if opts.InitialGuess != nil {
		if len(opts.InitialGuess) != rows {
			return nil, &sparse.DimMismatchError{
				Operation: "Solve", Rows: rows, Cols: cols,
				VecLen: len(opts.InitialGuess), Detail: "初值长度必须等于矩阵阶数",
			}
		}
		copy(x0, opts.InitialGuess)
	}

	// 5. 预处理器（开启预处理不改变原问题：仅以 M⁻¹ 改变迭代路径）。
	pc, err := NewPreconditioner(opts.Preconditioner, csr)
	if err != nil {
		var spe *SingularProblemError
		if errors.As(err, &spe) {
			res := &Result{
				Status:         StatusSingular,
				Method:         opts.Method,
				Criterion:      opts.Criterion.String(),
				Preconditioner: opts.Preconditioner,
				X:              append([]float64(nil), x0...),
				MaxIterations:  opts.MaxIterations,
				Tol:            opts.Tol,
				AbsTol:         opts.AbsTol,
				FailureReason:  spe.Error(),
			}
			return res, nil
		}
		return nil, err
	}

	res := &Result{
		Method:         opts.Method,
		Criterion:      opts.Criterion.String(),
		Preconditioner: opts.Preconditioner,
		MaxIterations:  opts.MaxIterations,
		Tol:            opts.Tol,
		AbsTol:         opts.AbsTol,
	}

	// 6. 分派迭代方法。
	switch opts.Method {
	case MethodCG:
		cgRun(csr, b, x0, opts, pc, res)
	case MethodBiCGSTAB:
		bicgRun(csr, b, x0, opts, pc, res)
	default:
		return nil, &sparse.FormatMismatchError{
			Operation: "Solve", Got: fmt.Sprintf("Method(%d)", opts.Method),
			Expect: "CG 或 BiCGSTAB", Detail: "未知求解方法",
		}
	}

	// 7. 可选的条件数估计（仅对收敛结果做 Lanczos 谱估计）。
	if opts.ConditionEstimate && res.Status == StatusConverged {
		res.ConditionEst = estimateConditionUpperBound(csr, opts)
	}
	return res, nil
}

// toCSR 将任意 sparse.Matrix 归一为内部使用的 *CSR。
func toCSR(A sparse.Matrix) (*sparse.CSR, error) {
	switch m := A.(type) {
	case *sparse.CSR:
		return m.ToCSR() // 返回校验过的拷贝
	case *sparse.CSC:
		return m.ToCSR()
	case *sparse.COO:
		return m.ToCSR()
	default:
		if c, ok := A.(sparse.Convertible); ok {
			return c.ToCSR()
		}
		return nil, &sparse.FormatMismatchError{
			Operation: "Solve", Got: fmt.Sprintf("%T", A),
			Expect: "*sparse.CSR / *sparse.CSC / *sparse.COO",
			Detail: "统一入口仅支持可转换的稀疏矩阵格式",
		}
	}
}

// detectStructuralSingularity 检查必然导致奇异的整行/整列全零结构。
func detectStructuralSingularity(A *sparse.CSR) error {
	n, m := A.Dims()
	if n != m {
		return &sparse.FormatMismatchError{
			Operation: "Solve", Got: fmt.Sprintf("%dx%d", n, m),
			Expect: "方阵", Detail: "迭代求解要求系数矩阵为方阵",
		}
	}
	for i := 0; i < n; i++ {
		if A.RowPtr[i] == A.RowPtr[i+1] {
			return &SingularProblemError{Reason: fmt.Sprintf(
				"第 %d 行整行为零，矩阵奇异，方程组无解或有无穷多解", i)}
		}
	}
	colSeen := make([]bool, m)
	for _, j := range A.ColInd {
		colSeen[j] = true
	}
	for j := 0; j < m; j++ {
		if !colSeen[j] {
			return &SingularProblemError{Reason: fmt.Sprintf(
				"第 %d 列整列为零，矩阵奇异，方程组无解或有无穷多解", j)}
		}
	}
	return nil
}

// withDefaults 填充未设置的选项并做输入校验。
func withDefaults(A sparse.Matrix, b []float64, opts Options) (Options, error) {
	rows, cols := A.Dims()
	if rows < 0 || cols < 0 {
		return opts, &sparse.DimMismatchError{
			Operation: "Solve", Rows: rows, Cols: cols, Detail: "矩阵维度非法"}
	}
	if rows != cols {
		return opts, &sparse.FormatMismatchError{
			Operation: "Solve", Got: fmt.Sprintf("%dx%d", rows, cols),
			Expect: "方阵", Detail: "迭代求解要求系数矩阵为方阵",
		}
	}
	if len(b) != rows {
		return opts, &sparse.DimMismatchError{
			Operation: "Solve", Rows: rows, Cols: cols, VecLen: len(b),
			Detail: "b 长度必须等于矩阵行数",
		}
	}
	for _, v := range b {
		if !isFinite(v) {
			return opts, &sparse.FormatMismatchError{
				Operation: "Solve", Got: "b 含 NaN/Inf", Expect: "有限实数右端项",
				Detail: "右端项必须全部为有限实数",
			}
		}
	}
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = rows
		if opts.MaxIterations < 1 {
			opts.MaxIterations = 1
		}
	}
	if opts.Tol <= 0 {
		opts.Tol = 1e-8
	}
	if opts.AbsTol <= 0 {
		opts.AbsTol = 1e-10
	}
	if opts.DivergenceFactor <= 0 {
		opts.DivergenceFactor = 1e8
	}
	return opts, nil
}
