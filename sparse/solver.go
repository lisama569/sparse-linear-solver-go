package sparse

import (
	"fmt"
	"math"
)

// Status 表示一次求解的最终状态，三者互斥。
type Status int

const (
	// Converged 在最大步数内满足收敛判据。
	Converged Status = iota
	// MaxIterationsReached 达到步数上限仍未收敛。
	MaxIterationsReached
	// Failed 数值发散或解算失效（如零主元、NaN/Inf、预处理器构造失败）。
	Failed
)

func (s Status) String() string {
	switch s {
	case Converged:
		return "converged"
	case MaxIterationsReached:
		return "max-iterations-reached"
	case Failed:
		return "failed"
	}
	return "unknown"
}

// Config 控制迭代求解过程。
type Config struct {
	Tolerance      float64        // 相对残差收敛判据 |r|/|b| <= Tolerance
	MaxIterations  int            // 最大迭代步数
	Preconditioner Preconditioner // 可选，nil 表示不使用
}

// 缺省配置。
const (
	DefaultTolerance     = 1e-10
	DefaultMaxIterations = 10000
)

func (c *Config) withDefaults() Config {
	out := *c
	if out.Tolerance <= 0 {
		out.Tolerance = DefaultTolerance
	}
	if out.MaxIterations <= 0 {
		out.MaxIterations = DefaultMaxIterations
	}
	return out
}

// Result 是一次求解的完整、可解释的结果。
type Result struct {
	Status     Status
	X          []float64 // 仅在 Converged 时保证有意义
	Iterations int
	Residual   float64 // 最终相对残差 |r|/|b|
	Reason     string  // 失败或未收敛时的可读原因
}

// Method 是迭代求解方法的统一接口。
type Method interface {
	Name() string
	Solve(a Matrix, b []float64, cfg Config) Result
}

// Solve 是统一入口：对任意 Matrix 用指定方法求解 A*x = b。
// 维度不合法、右端含 NaN/Inf 等输入问题直接以 Failed 返回，不进入迭代。
func Solve(a Matrix, b []float64, m Method, cfg Config) Result {
	rows, cols := a.Dims()
	if rows != cols {
		return Result{Status: Failed, Reason: fmt.Sprintf("%v: need square matrix, got %dx%d", ErrDimensionMismatch, rows, cols)}
	}
	if len(b) != rows {
		return Result{Status: Failed, Reason: fmt.Sprintf("%v: len(b)=%d, matrix order=%d", ErrDimensionMismatch, len(b), rows)}
	}
	for i, v := range b {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return Result{Status: Failed, Reason: fmt.Sprintf("b[%d] is %v, refuse to solve", i, v)}
		}
	}
	return m.Solve(a, b, cfg.withDefaults())
}

// ---- 各方法共用的向量辅助 ----

func dot(x, y []float64) float64 {
	var s float64
	for i := range x {
		s += x[i] * y[i]
	}
	return s
}

func norm2(x []float64) float64 { return math.Sqrt(dot(x, x)) }

// hasNaNOrInf 检测数值发散。
func hasNaNOrInf(x []float64) bool {
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// residualOf 计算 r = b - A*x 及其相对范数 |r|/|b|（|b|=0 时用 |r|）。
func residualOf(a Matrix, b, x []float64) (relRes float64, err error) {
	ax, err := a.MatVec(x)
	if err != nil {
		return 0, err
	}
	r := make([]float64, len(b))
	for i := range b {
		r[i] = b[i] - ax[i]
	}
	denom := norm2(b)
	if denom == 0 {
		denom = 1
	}
	return norm2(r) / denom, nil
}
