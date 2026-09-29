// Package solver 提供稀疏线性方程组 A*x = b 的统一迭代求解入口，
// 支持多种迭代方法、收敛判据配置与可选预处理器。
package solver

import "fmt"

// Status 是一次求解的确定终态。
type Status int

const (
	// StatusConverged 表示残差满足配置的收敛判据。
	StatusConverged Status = iota
	// StatusMaxIteration 表示达到最大迭代步数仍未收敛（结果不可作为解使用）。
	StatusMaxIteration
	// StatusDiverged 表示数值发散或解算失效（NaN/Inf、残差爆炸等）。
	StatusDiverged
	// StatusSingular 表示检测到方程组奇异或明显病态，无法给出有意义的解。
	StatusSingular
)

// String 返回状态的中文可读描述。
func (s Status) String() string {
	switch s {
	case StatusConverged:
		return "已收敛"
	case StatusMaxIteration:
		return "达到步数上限仍未收敛"
	case StatusDiverged:
		return "数值发散或解算失效"
	case StatusSingular:
		return "奇异或明显病态，解不存在或不唯一"
	default:
		return fmt.Sprintf("未知状态(%d)", int(s))
	}
}

// Method 标识迭代求解方法。
type Method int

const (
	// MethodCG 为共轭梯度法，要求矩阵对称正定。
	MethodCG Method = iota
	// MethodBiCGSTAB 为稳定双共轭梯度法，适用于一般非对称方阵。
	MethodBiCGSTAB
)

// String 返回方法名称。
func (m Method) String() string {
	switch m {
	case MethodCG:
		return "CG"
	case MethodBiCGSTAB:
		return "BiCGSTAB"
	default:
		return fmt.Sprintf("未知方法(%d)", int(m))
	}
}

// ResidualCriterion 选择残差收敛判据。
type ResidualCriterion int

const (
	// RelativeResidual 使用 ‖b-Ax‖₂/‖b‖₂ <= Tol 的相对残差判据。
	RelativeResidual ResidualCriterion = iota
	// AbsoluteResidual 使用 ‖b-Ax‖₂ <= AbsTol 的绝对残差判据。
	AbsoluteResidual
	// BothResidual 要求相对残差与绝对残差同时满足（默认）。
	BothResidual
)

// String 返回判据名称。
func (c ResidualCriterion) String() string {
	switch c {
	case RelativeResidual:
		return "相对残差‖r‖₂/‖b‖₂"
	case AbsoluteResidual:
		return "绝对残差‖r‖₂"
	case BothResidual:
		return "相对与绝对残差同时满足"
	default:
		return fmt.Sprintf("未知判据(%d)", int(c))
	}
}

// PrecondType 标识预处理器类型。
type PrecondType int

const (
	// PrecondNone 不使用预处理。
	PrecondNone PrecondType = iota
	// PrecondJacobi 使用（对角）雅可比预处理器。
	PrecondJacobi
	// PrecondILU0 使用零填充不完全 LU 分解（ILU(0)）。
	PrecondILU0
)

// String 返回预处理器名称。
func (p PrecondType) String() string {
	switch p {
	case PrecondNone:
		return "无预处理"
	case PrecondJacobi:
		return "Jacobi"
	case PrecondILU0:
		return "ILU(0)"
	default:
		return fmt.Sprintf("未知预处理器(%d)", int(p))
	}
}

// Options 配置一次求解的方法、收敛判据与最大迭代步数。
type Options struct {
	Method            Method            // 求解方法，默认 CG
	MaxIterations     int               // 最大迭代步数，<=0 时取矩阵阶数
	Tol               float64           // 相对残差容差，<=0 时取 1e-8
	AbsTol            float64           // 绝对残差容差，<=0 时取 1e-10
	Criterion         ResidualCriterion // 收敛判据，默认 BothResidual
	Preconditioner    PrecondType       // 预处理器，默认不使用
	InitialGuess      []float64         // 可选初值，长度须等于矩阵阶数；nil 表示零向量
	DivergenceFactor  float64           // 残差相对初值增长超过该倍数判为发散，<=0 时取 1e8
	ConditionEstimate bool              // 是否在结果中报告估计的条件数上界
}

// Result 是一次求解的可解释、可复现结论。
type Result struct {
	Status           Status      // 终态：收敛 / 步数上限 / 发散 / 奇异
	Method           Method      // 实际使用的方法
	Criterion        string      // 实际使用的收敛判据描述
	Preconditioner   PrecondType // 实际使用的预处理器
	X                []float64   // 最终迭代向量（仅在 Converged 时有意义）
	Iterations       int         // 实际执行的迭代步数
	FinalResidual    float64     // 最终残差 2-范数 ‖b-Ax‖₂
	RelativeResidual float64     // 最终相对残差 ‖b-Ax‖₂/‖b‖₂
	MaxIterations    int         // 本次配置的步数上限
	Tol              float64     // 本次配置的相对容差
	AbsTol           float64     // 本次配置的绝对容差
	ConditionEst     float64     // 条件数上界估计（未请求或不可估时为 0）
	FailureReason    string      // 非收敛时可区分的失败原因
}

// Converged 报告本次求解是否成功收敛。
func (r *Result) Converged() bool {
	return r != nil && r.Status == StatusConverged
}

// Summary 返回单行的可读结论，便于写入测试日志。
func (r *Result) Summary() string {
	if r == nil {
		return "<nil result>"
	}
	s := fmt.Sprintf("方法=%s 预处理器=%s 判据=%s 状态=%s 迭代=%d/%d 残差=%.3e 相对残差=%.3e",
		r.Method, r.Preconditioner, r.Criterion, r.Status,
		r.Iterations, r.MaxIterations, r.FinalResidual, r.RelativeResidual)
	if r.FailureReason != "" {
		s += " 原因=" + r.FailureReason
	}
	return s
}
