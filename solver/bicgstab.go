package solver

import (
	"fmt"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// bicgRun 执行预条件稳定双共轭梯度法（左预处理形式 BiCGSTAB）。
// 适用于一般非对称方阵；除零、NaN/Inf、残差爆炸均判定为发散/失效。
func bicgRun(A *sparse.CSR, b []float64, x0 []float64, opts Options, pc Preconditioner, res *Result) {
	n := len(b)
	x := append([]float64(nil), x0...)
	res.X = x

	ax, err := A.MatVec(x)
	if err != nil {
		res.Status = StatusDiverged
		res.FailureReason = "矩阵向量乘失败: " + err.Error()
		return
	}
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = b[i] - ax[i]
	}
	rhat := append([]float64(nil), r...) // 影子向量 rhat0 = r0
	bnorm := norm2(b)
	r0norm := norm2(r)

	if ok, rel := checkConvergence(r0norm, bnorm, opts); ok {
		res.Status = StatusConverged
		res.Iterations = 0
		res.FinalResidual = r0norm
		res.RelativeResidual = rel
		return
	}

	rho := 1.0
	alpha := 1.0
	w := 1.0
	p := make([]float64, n)
	v := make([]float64, n)

	finishFailure := func(k int, rn float64, status Status, format string, args ...any) {
		res.Status = status
		res.Iterations = k
		res.FinalResidual = rn
		res.RelativeResidual = relOrInf(rn, bnorm)
		res.FailureReason = fmt.Sprintf(format, args...)
	}

	for k := 1; k <= opts.MaxIterations; k++ {
		rhoNew := dot(rhat, r)
		if !isFinite(rhoNew) {
			finishFailure(k, norm2(r), StatusDiverged,
				"第 %d 步标量 rho 为 NaN/Inf，数值发散", k)
			return
		}
		if rhoNew == 0 {
			// BiCG 项崩溃：影子向量与残差正交，方法对该问题失效。
			finishFailure(k, norm2(r), StatusDiverged,
				"第 %d 步 rhat·r=0，BiCGSTAB 崩溃（breakdown），解算失效；可尝试更换初值或预处理器", k)
			return
		}
		if k == 1 {
			copy(p, r)
		} else {
			beta := (rhoNew / rho) * (alpha / w)
			if !isFinite(beta) {
				finishFailure(k, norm2(r), StatusDiverged,
					"第 %d 步 beta 为 NaN/Inf，数值发散", k)
				return
			}
			for i := 0; i < n; i++ {
				p[i] = r[i] + beta*(p[i]-w*v[i])
			}
		}

		// 左预处理： p̂ = M⁻¹ p，v = A p̂
		phat, err := pc.Apply(p)
		if err != nil {
			res.Status = StatusSingular
			res.FailureReason = "预处理器失效: " + err.Error()
			return
		}
		v, err = A.MatVec(phat)
		if err != nil {
			res.Status = StatusDiverged
			res.FailureReason = "矩阵向量乘失败: " + err.Error()
			return
		}
		rhatv := dot(rhat, v)
		if !isFinite(rhatv) || rhatv == 0 {
			finishFailure(k, norm2(r), StatusDiverged,
				"第 %d 步 rhat·v=%.6e，BiCGSTAB 崩溃，解算失效", k, rhatv)
			return
		}
		alpha = rhoNew / rhatv
		if !isFinite(alpha) {
			finishFailure(k, norm2(r), StatusDiverged,
				"第 %d 步 alpha 为 NaN/Inf，数值发散", k)
			return
		}
		// s = r - alpha*v
		s := make([]float64, n)
		for i := 0; i < n; i++ {
			s[i] = r[i] - alpha*v[i]
		}
		sn := norm2(s)
		if hasNonFinite(s) {
			finishFailure(k, sn, StatusDiverged, "第 %d 步出现 NaN/Inf，数值发散", k)
			return
		}
		if ok, rel := checkConvergence(sn, bnorm, opts); ok {
			for i := 0; i < n; i++ {
				x[i] += alpha * phat[i]
			}
			res.Status = StatusConverged
			res.Iterations = k
			res.FinalResidual = sn
			res.RelativeResidual = rel
			return
		}

		// ŝ = M⁻¹ s，t = A ŝ
		shat, err := pc.Apply(s)
		if err != nil {
			res.Status = StatusSingular
			res.FailureReason = "预处理器失效: " + err.Error()
			return
		}
		t, err := A.MatVec(shat)
		if err != nil {
			res.Status = StatusDiverged
			res.FailureReason = "矩阵向量乘失败: " + err.Error()
			return
		}
		tt := dot(t, t)
		if !isFinite(tt) || hasNonFinite(t) {
			finishFailure(k, sn, StatusDiverged, "第 %d 步 t 出现 NaN/Inf，数值发散", k)
			return
		}
		if tt == 0 {
			// t=0：要么已得到精确解，要么停滞。用真实残差复核。
			for i := 0; i < n; i++ {
				x[i] += alpha * phat[i]
			}
			ax, _ = A.MatVec(x)
			rn := 0.0
			for i := 0; i < n; i++ {
				d := b[i] - ax[i]
				rn += d * d
			}
			rn = sqrt(rn)
			if ok, rel := checkConvergence(rn, bnorm, opts); ok {
				res.Status = StatusConverged
				res.Iterations = k
				res.FinalResidual = rn
				res.RelativeResidual = rel
				return
			}
			finishFailure(k, rn, StatusDiverged,
				"第 %d 步 t=0 且残差未达标，BiCGSTAB 停滞失效", k)
			return
		}
		w = dot(t, s) / tt
		if !isFinite(w) {
			finishFailure(k, sn, StatusDiverged, "第 %d 步 omega 为 NaN/Inf，数值发散", k)
			return
		}
		// x ← x + alpha p̂ + w ŝ
		for i := 0; i < n; i++ {
			x[i] += alpha*phat[i] + w*shat[i]
		}
		// r = s - w t
		for i := 0; i < n; i++ {
			r[i] = s[i] - w*t[i]
		}
		rn := norm2(r)
		if hasNonFinite(x) || !isFinite(rn) {
			finishFailure(k, rn, StatusDiverged, "第 %d 步解或残差出现 NaN/Inf，数值发散", k)
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
		if r0norm > 0 && rn > opts.DivergenceFactor*r0norm {
			finishFailure(k, rn, StatusDiverged,
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
		rho = rhoNew
		if w == 0 {
			finishFailure(k, rn, StatusDiverged, "第 %d 步 omega=0，BiCGSTAB 崩溃", k)
			return
		}
	}
	res.Status = StatusMaxIteration
	res.FailureReason = "迭代循环异常结束：达到步数上限"
}

// relOrInf 在 bnorm=0 时返回 Inf，保持失败信息可读。
func relOrInf(rn, bnorm float64) float64 {
	if bnorm == 0 {
		if rn == 0 {
			return 0
		}
		return inf()
	}
	return rn / bnorm
}
