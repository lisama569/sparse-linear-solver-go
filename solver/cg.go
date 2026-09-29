package solver

import (
	"fmt"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// cgRun 执行预条件共轭梯度法（PCG）。
// A 应为对称正定矩阵；若迭代中检测到非正定（p·Ap 非正）则给出发散/失效结论。
func cgRun(A *sparse.CSR, b []float64, x0 []float64, opts Options, pc Preconditioner, res *Result) {
	n := len(b)
	x := append([]float64(nil), x0...)
	res.X = x

	ax, err := A.MatVec(x)
	if err != nil {
		res.Status = StatusDiverged
		res.FailureReason = "矩阵向量乘失败: " + err.Error()
		return
	}
	r := make([]float64, n) // r = b - A x
	for i := 0; i < n; i++ {
		r[i] = b[i] - ax[i]
	}
	bnorm := norm2(b)
	r0norm := norm2(r)

	z, err := pc.Apply(r)
	if err != nil {
		res.Status = StatusSingular
		res.FailureReason = "预处理器失效: " + err.Error()
		return
	}
	p := append([]float64(nil), z...)
	rz := dot(r, z)

	// 初值即满足判据。
	if ok, rel := checkConvergence(r0norm, bnorm, opts); ok {
		res.Status = StatusConverged
		res.Iterations = 0
		res.FinalResidual = r0norm
		res.RelativeResidual = rel
		return
	}

	for k := 1; k <= opts.MaxIterations; k++ {
		ap, err := A.MatVec(p)
		if err != nil {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = "矩阵向量乘失败: " + err.Error()
			return
		}
		pap := dot(p, ap)
		if !isFinite(pap) || hasNonFinite(ap) {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf("第 %d 步出现 NaN/Inf，数值发散或解算失效", k)
			return
		}
		if pap <= 0 {
			// CG 的数学前提被破坏：矩阵非 SPD（或预处理器不定）。
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf(
				"第 %d 步 p·A p=%.6e <= 0，矩阵非对称正定，CG 不适用", k, pap)
			return
		}
		alpha := rz / pap
		if !isFinite(alpha) {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf("第 %d 步步长为 NaN/Inf，解算失效", k)
			return
		}
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
		}
		// 用真实残差 r = b-Ax 做判据，避免长迭代中的递推漂移。
		ax, err = A.MatVec(x)
		if err != nil {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = "矩阵向量乘失败: " + err.Error()
			return
		}
		for i := 0; i < n; i++ {
			r[i] = b[i] - ax[i]
		}
		rn := norm2(r)
		if hasNonFinite(x) || !isFinite(rn) {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf("第 %d 步解或残差出现 NaN/Inf，数值发散", k)
			return
		}

		ok, rel := checkConvergence(rn, bnorm, opts)
		if ok {
			res.Status = StatusConverged
			res.Iterations = k
			res.FinalResidual = rn
			res.RelativeResidual = rel
			return
		}
		// 相对初始残差爆炸式增长判为发散。
		if r0norm > 0 && rn > opts.DivergenceFactor*r0norm {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf(
				"第 %d 步残差 %.3e 相对初值 %.3e 增长超过 %.0e 倍，数值发散",
				k, rn, r0norm, opts.DivergenceFactor)
			return
		}
		if k == opts.MaxIterations {
			res.Status = StatusMaxIteration
			res.Iterations = k
			res.FinalResidual = rn
			res.RelativeResidual = rel
			res.FailureReason = fmt.Sprintf(
				"达到最大迭代步数 %d 仍未满足判据（相对残差 %.3e，容差 %.2e）",
				k, rel, opts.Tol)
			return
		}

		z, err = pc.Apply(r)
		if err != nil {
			res.Iterations = k
			res.Status = StatusSingular
			res.FailureReason = "预处理器失效: " + err.Error()
			return
		}
		rzNew := dot(r, z)
		if rz == 0 || !isFinite(rzNew) {
			res.Iterations = k
			res.Status = StatusDiverged
			res.FailureReason = fmt.Sprintf("第 %d 步预条件内积退化，解算失效", k)
			return
		}
		beta := rzNew / rz
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rz = rzNew
	}
	// 理论上不会走到这里，兜底标记为步数上限。
	res.Status = StatusMaxIteration
	res.FailureReason = "迭代循环异常结束：达到步数上限"
}
