package sparse

import (
	"fmt"
	"math"
)

// 相对残差超过该倍数即判定为数值发散。
const divergenceFactor = 1e6

// toCSRView 把任意 Matrix 转成 CSR 视图，供需要行访问的算法使用。
func toCSRView(a Matrix) (*CSR, error) {
	switch m := a.(type) {
	case *CSR:
		return m, nil
	case *COO:
		return m.ToCSR(), nil
	case *CSC:
		return m.ToCSR(), nil
	default:
		return nil, fmt.Errorf("unsupported matrix type %T", a)
	}
}

// diagonalOf 提取对角元，任一为零时返回 ErrZeroPivot。
func diagonalOf(a Matrix, n int) ([]float64, error) {
	csr, err := toCSRView(a)
	if err != nil {
		return nil, err
	}
	diag := make([]float64, n)
	for i := 0; i < csr.Rows; i++ {
		for k := csr.RowPtr[i]; k < csr.RowPtr[i+1]; k++ {
			if csr.ColIdx[k] == i {
				diag[i] += csr.Values[k]
			}
		}
	}
	for i, d := range diag {
		if d == 0 || math.IsNaN(d) || math.IsInf(d, 0) {
			return nil, fmt.Errorf("%w at diagonal %d (value %v)", ErrZeroPivot, i, d)
		}
	}
	return diag, nil
}

// Jacobi 迭代法，要求对角元非零。
type Jacobi struct{}

func (Jacobi) Name() string { return "jacobi" }

func (Jacobi) Solve(a Matrix, b []float64, cfg Config) Result {
	n := len(b)
	diag, err := diagonalOf(a, n)
	if err != nil {
		return Result{Status: Failed, Reason: "jacobi: " + err.Error()}
	}
	x := make([]float64, n)
	next := make([]float64, n)
	res0, err := residualOf(a, b, x)
	if err != nil {
		return Result{Status: Failed, Reason: "jacobi: " + err.Error()}
	}
	if res0 <= cfg.Tolerance {
		return Result{Status: Converged, X: x, Iterations: 0, Residual: res0}
	}
	for it := 1; it <= cfg.MaxIterations; it++ {
		ax, err := a.MatVec(x)
		if err != nil {
			return Result{Status: Failed, Reason: "jacobi: " + err.Error()}
		}
		for i := 0; i < n; i++ {
			next[i] = x[i] + (b[i]-ax[i])/diag[i]
		}
		x, next = next, x
		if hasNaNOrInf(x) {
			return Result{Status: Failed, Iterations: it, Reason: "jacobi: numerical divergence (NaN/Inf in iterate)"}
		}
		rel, err := residualOf(a, b, x)
		if err != nil {
			return Result{Status: Failed, Reason: "jacobi: " + err.Error()}
		}
		if rel <= cfg.Tolerance {
			return Result{Status: Converged, X: x, Iterations: it, Residual: rel}
		}
		if rel > divergenceFactor*res0 {
			return Result{Status: Failed, Iterations: it, Residual: rel, Reason: "jacobi: residual diverged"}
		}
	}
	rel, _ := residualOf(a, b, x)
	return Result{Status: MaxIterationsReached, X: x, Iterations: cfg.MaxIterations, Residual: rel,
		Reason: fmt.Sprintf("jacobi: relative residual %.3e > tolerance %.3e after %d iterations", rel, cfg.Tolerance, cfg.MaxIterations)}
}

// GaussSeidel 迭代法（内部使用 CSR 行访问）。
type GaussSeidel struct{}

func (GaussSeidel) Name() string { return "gauss-seidel" }

func (GaussSeidel) Solve(a Matrix, b []float64, cfg Config) Result {
	n := len(b)
	csr, err := toCSRView(a)
	if err != nil {
		return Result{Status: Failed, Reason: "gauss-seidel: " + err.Error()}
	}
	diag, err := diagonalOf(a, n)
	if err != nil {
		return Result{Status: Failed, Reason: "gauss-seidel: " + err.Error()}
	}
	x := make([]float64, n)
	res0, err := residualOf(a, b, x)
	if err != nil {
		return Result{Status: Failed, Reason: "gauss-seidel: " + err.Error()}
	}
	if res0 <= cfg.Tolerance {
		return Result{Status: Converged, X: x, Iterations: 0, Residual: res0}
	}
	for it := 1; it <= cfg.MaxIterations; it++ {
		for i := 0; i < n; i++ {
			sigma := 0.0
			for k := csr.RowPtr[i]; k < csr.RowPtr[i+1]; k++ {
				j := csr.ColIdx[k]
				if j != i {
					sigma += csr.Values[k] * x[j]
				}
			}
			x[i] = (b[i] - sigma) / diag[i]
		}
		if hasNaNOrInf(x) {
			return Result{Status: Failed, Iterations: it, Reason: "gauss-seidel: numerical divergence (NaN/Inf in iterate)"}
		}
		rel, err := residualOf(a, b, x)
		if err != nil {
			return Result{Status: Failed, Reason: "gauss-seidel: " + err.Error()}
		}
		if rel <= cfg.Tolerance {
			return Result{Status: Converged, X: x, Iterations: it, Residual: rel}
		}
		if rel > divergenceFactor*res0 {
			return Result{Status: Failed, Iterations: it, Residual: rel, Reason: "gauss-seidel: residual diverged"}
		}
	}
	rel, _ := residualOf(a, b, x)
	return Result{Status: MaxIterationsReached, X: x, Iterations: cfg.MaxIterations, Residual: rel,
		Reason: fmt.Sprintf("gauss-seidel: relative residual %.3e > tolerance %.3e after %d iterations", rel, cfg.Tolerance, cfg.MaxIterations)}
}

// CG 共轭梯度法，要求 A 对称正定；可选预处理器（PCG）。
type CG struct{}

func (CG) Name() string { return "cg" }

func (CG) Solve(a Matrix, b []float64, cfg Config) Result {
	n := len(b)
	if cfg.Preconditioner != nil {
		if err := cfg.Preconditioner.Setup(a); err != nil {
			return Result{Status: Failed, Reason: "cg: preconditioner setup: " + err.Error()}
		}
	}
	x := make([]float64, n)
	// r = b - A*x = b（x 从零向量出发）
	r := make([]float64, n)
	copy(r, b)
	denom := norm2(b)
	if denom == 0 {
		denom = 1
	}
	applyM := func(v []float64) ([]float64, error) {
		if cfg.Preconditioner == nil {
			out := make([]float64, len(v))
			copy(out, v)
			return out, nil
		}
		return cfg.Preconditioner.Apply(v)
	}
	z, err := applyM(r)
	if err != nil {
		return Result{Status: Failed, Reason: "cg: " + err.Error()}
	}
	p := make([]float64, n)
	copy(p, z)
	rz := dot(r, z)
	res0 := norm2(r) / denom
	if res0 <= cfg.Tolerance {
		return Result{Status: Converged, X: x, Iterations: 0, Residual: res0}
	}
	for it := 1; it <= cfg.MaxIterations; it++ {
		ap, err := a.MatVec(p)
		if err != nil {
			return Result{Status: Failed, Reason: "cg: " + err.Error()}
		}
		pap := dot(p, ap)
		if pap <= 0 || math.IsNaN(pap) || math.IsInf(pap, 0) {
			return Result{Status: Failed, Iterations: it,
				Reason: fmt.Sprintf("cg: breakdown p^T A p = %v (%v)", pap, ErrNotSymmetric)}
		}
		alpha := rz / pap
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
			r[i] -= alpha * ap[i]
		}
		if hasNaNOrInf(x) {
			return Result{Status: Failed, Iterations: it, Reason: "cg: numerical divergence (NaN/Inf in iterate)"}
		}
		rel := norm2(r) / denom
		if rel <= cfg.Tolerance {
			return Result{Status: Converged, X: x, Iterations: it, Residual: rel}
		}
		if rel > divergenceFactor*res0 {
			return Result{Status: Failed, Iterations: it, Residual: rel, Reason: "cg: residual diverged"}
		}
		z, err = applyM(r)
		if err != nil {
			return Result{Status: Failed, Reason: "cg: " + err.Error()}
		}
		rzNew := dot(r, z)
		beta := rzNew / rz
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rz = rzNew
	}
	rel, _ := residualOf(a, b, x)
	return Result{Status: MaxIterationsReached, X: x, Iterations: cfg.MaxIterations, Residual: rel,
		Reason: fmt.Sprintf("cg: relative residual %.3e > tolerance %.3e after %d iterations", rel, cfg.Tolerance, cfg.MaxIterations)}
}
