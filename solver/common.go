package solver

import "math"

// isFinite 判断 x 是否为有限实数（非 NaN、非 Inf）。
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// dot 计算两个等长向量的内积。
func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// norm2 计算向量 2-范数。
func norm2(a []float64) float64 {
	return math.Sqrt(dot(a, a))
}

// scaledVectors 报告切片中是否出现任何非有限元素。
func hasNonFinite(v []float64) bool {
	for _, x := range v {
		if !isFinite(x) {
			return true
		}
	}
	return false
}

// converged 按配置判据判断残差是否达标。
// 返回是否收敛以及实际采用的相对/绝对残差值。
func checkConvergence(rn, bnorm float64, opts Options) (bool, float64) {
	rel := 0.0
	if bnorm > 0 {
		rel = rn / bnorm
	} else {
		// b=0 时相对残差没有意义：x=0 必为解，残差为 0 即绝对收敛。
		if rn == 0 {
			rel = 0
		} else {
			rel = math.Inf(1)
		}
	}
	switch opts.Criterion {
	case RelativeResidual:
		return rn <= opts.AbsTol || rel <= opts.Tol, rel
	case AbsoluteResidual:
		return rn <= opts.AbsTol, rel
	case BothResidual:
		fallthrough
	default:
		// 同时满足相对与绝对容差；bnorm=0 时只要求绝对残差。
		if bnorm == 0 {
			return rn <= opts.AbsTol, rel
		}
		return rel <= opts.Tol && rn <= opts.AbsTol, rel
	}
}

// sqrt 是 math.Sqrt 的简短别名。
func sqrt(x float64) float64 { return math.Sqrt(x) }

// inf 返回正无穷。
func inf() float64 { return math.Inf(1) }
