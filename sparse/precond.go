package sparse

import (
	"fmt"
	"math"
)

// Preconditioner 近似 M ≈ A，通过 Apply 求解 M*z = r。
// 预处理只改变迭代路径，不改变原问题的解。
type Preconditioner interface {
	Name() string
	// Setup 从矩阵构造预处理器，奇异（零对角等）时返回错误。
	Setup(a Matrix) error
	// Apply 求解 M*z = r，返回 z。
	Apply(r []float64) ([]float64, error)
}

// JacobiPreconditioner 使用 A 的对角线，要求对角元全部非零。
type JacobiPreconditioner struct {
	diag []float64
}

func (p *JacobiPreconditioner) Name() string { return "jacobi" }

func (p *JacobiPreconditioner) Setup(a Matrix) error {
	rows, cols := a.Dims()
	if rows != cols {
		return fmt.Errorf("jacobi preconditioner: %w: need square matrix, got %dx%d", ErrDimensionMismatch, rows, cols)
	}
	p.diag = make([]float64, rows)
	// 通过 COO 视图收集对角元，与具体存储格式解耦。
	var coo *COO
	switch m := a.(type) {
	case *COO:
		coo = m
	case *CSR:
		coo = m.ToCOO()
	case *CSC:
		coo = m.ToCOO()
	default:
		return fmt.Errorf("jacobi preconditioner: unsupported matrix type %T", a)
	}
	for _, e := range coo.Entries {
		if e.Row == e.Col {
			p.diag[e.Row] += e.Value
		}
	}
	for i, d := range p.diag {
		if d == 0 || math.IsNaN(d) || math.IsInf(d, 0) {
			return fmt.Errorf("jacobi preconditioner: %w at diagonal %d (value %v)", ErrZeroPivot, i, d)
		}
	}
	return nil
}

func (p *JacobiPreconditioner) Apply(r []float64) ([]float64, error) {
	if len(r) != len(p.diag) {
		return nil, fmt.Errorf("jacobi preconditioner: %w: vector length %d, matrix order %d", ErrDimensionMismatch, len(r), len(p.diag))
	}
	z := make([]float64, len(r))
	for i := range r {
		z[i] = r[i] / p.diag[i]
	}
	return z, nil
}
