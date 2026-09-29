package solver

import (
	"math"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// estimateConditionUpperBound 用 Lanczos 过程粗略估计 A 的谱条件数上界
// κ₂(A) ≈ |λ_max|/|λ_min|，用于在收敛后提示问题的病态程度。
// 该值仅为启发式估计，不参与收敛判定；无法估计时返回 0。
//
// 对对称矩阵，Lanczos 的 Ritz 值逐步逼近 A 的极值特征值；对非对称矩阵，
// 得到的是 AᵀA 的近似（基于两次 MatVec），因此估计值一律视为上界提示。
func estimateConditionUpperBound(A *sparse.CSR, opts Options) float64 {
	n, _ := A.Dims()
	steps := opts.MaxIterations
	if steps > n {
		steps = n
	}
	if steps < 2 {
		return 0
	}
	// 对称判定：A ≈ Aᵀ（按结构与数值）。
	symmetric := isSymmetric(A)

	if symmetric {
		return lanczosSymmetric(A, steps)
	}
	return lanczosNormal(A, steps)
}

// isSymmetric 检查 CSR 的稀疏结构与数值是否对称。
func isSymmetric(A *sparse.CSR) bool {
	n := A.Rows
	for i := 0; i < n; i++ {
		for k := A.RowPtr[i]; k < A.RowPtr[i+1]; k++ {
			j := A.ColInd[k]
			v := A.Data[k]
			// 查找 (j,i)。
			found := false
			for qk := A.RowPtr[j]; qk < A.RowPtr[j+1]; qk++ {
				if A.ColInd[qk] == i {
					if math.Abs(A.Data[qk]-v) > 1e-12*(1+math.Abs(v)) {
						return false
					}
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
	}
	return true
}

// lanczosSymmetric 对对称 A 做 Lanczos，返回 max|θ|/min|θ| 的 Ritz 估计。
func lanczosSymmetric(A *sparse.CSR, steps int) float64 {
	n := A.Rows
	q := make([][]float64, steps+1)
	q[0] = make([]float64, n)
	q[1] = make([]float64, n)
	q[1][0] = 1 // 起始向量 e0
	alpha := make([]float64, steps)
	beta := make([]float64, steps) // beta[1..]

	for j := 0; j < steps; j++ {
		w, err := A.MatVec(q[j+1])
		if err != nil || hasNonFinite(w) {
			return 0
		}
		alpha[j] = dot(w, q[j+1])
		if j > 0 {
			for i := 0; i < n; i++ {
				w[i] -= alpha[j]*q[j+1][i] + beta[j]*q[j][i]
			}
		} else {
			for i := 0; i < n; i++ {
				w[i] -= alpha[j] * q[j+1][i]
			}
		}
		// 全重正交化，抑制数值上的虚假特征值。
		for s := 0; s <= j+1; s++ {
			c := dot(w, q[s])
			for i := 0; i < n; i++ {
				w[i] -= c * q[s][i]
			}
		}
		bn := norm2(w)
		if bn < 1e-14 || j+1 == steps {
			return ritzCondition(alpha[:j+1], beta[1:j+1])
		}
		beta[j+1] = bn
		if j+2 <= steps {
			q[j+2] = make([]float64, n)
			for i := 0; i < n; i++ {
				q[j+2][i] = w[i] / bn
			}
		}
	}
	return ritzCondition(alpha, beta[1:])
}

// ritzCondition 由三对角 T 的主对角 alpha 与次对角 beta 估计 κ。
func ritzCondition(alpha, beta []float64) float64 {
	if len(alpha) == 0 {
		return 0
	}
	// 用 Gershgorin 圆盘给极值特征值一个确定性的界（不依赖 QR 迭代）。
	lmin, lmax := alpha[0], alpha[0]
	for i := range alpha {
		lo, hi := alpha[i], alpha[i]
		if i > 0 {
			lo -= math.Abs(beta[i-1])
			hi += math.Abs(beta[i-1])
		}
		if i < len(beta) {
			lo -= math.Abs(beta[i])
			hi += math.Abs(beta[i])
		}
		if lo < lmin {
			lmin = lo
		}
		if hi > lmax {
			lmax = hi
		}
	}
	if lmin <= 0 {
		// 非正定或界跨过零点，无法给 SPD 条件数；退化为极值绝对值之比。
		lmin = math.Abs(lmin)
		if lmax < lmin {
			lmax = lmin
		}
	}
	if lmin < 1e-300 || !isFinite(lmin) {
		return math.Inf(1)
	}
	return math.Abs(lmax) / math.Abs(lmin)
}

// lanczosNormal 对一般矩阵估计 AᵀA 的极值，开方得到 κ₂(A) 的提示。
func lanczosNormal(A *sparse.CSR, steps int) float64 {
	n := A.Rows
	q := make([]float64, n)
	q[0] = 1
	qPrev := make([]float64, n)
	alpha := make([]float64, 0, steps)
	beta := make([]float64, 0, steps)

	ata := func(v []float64) []float64 {
		av, err := A.MatVec(v)
		if err != nil {
			return nil
		}
		// Aᵀ(av)：借助 CSC 或直接计算。
		atv := make([]float64, n)
		for i := 0; i < n; i++ {
			for k := A.RowPtr[i]; k < A.RowPtr[i+1]; k++ {
				atv[A.ColInd[k]] += A.Data[k] * av[i]
			}
		}
		return atv
	}

	for j := 0; j < steps; j++ {
		w := ata(q)
		if w == nil || hasNonFinite(w) {
			break
		}
		a := dot(w, q)
		alpha = append(alpha, a)
		if j > 0 {
			b := beta[j-1]
			for i := 0; i < n; i++ {
				w[i] -= a*q[i] + b*qPrev[i]
			}
		} else {
			for i := 0; i < n; i++ {
				w[i] -= a * q[i]
			}
		}
		bn := norm2(w)
		if bn < 1e-14 {
			break
		}
		beta = append(beta, bn)
		qPrev = append([]float64(nil), q...)
		for i := 0; i < n; i++ {
			q[i] = w[i] / bn
		}
	}
	// AᵀA 半正定，其特征值界为 lmin,lmax；κ(A)=sqrt(lmax/lmin)。
	if len(alpha) == 0 {
		return 0
	}
	lmin, lmax := alpha[0], alpha[0]
	for i := range alpha {
		lo, hi := alpha[i], alpha[i]
		if i > 0 {
			lo -= math.Abs(beta[i-1])
			hi += math.Abs(beta[i-1])
		}
		if i < len(beta) {
			lo -= math.Abs(beta[i])
			hi += math.Abs(beta[i])
		}
		if lo < 0 {
			lo = 0
		}
		if lo < lmin {
			lmin = lo
		}
		if hi > lmax {
			lmax = hi
		}
	}
	if lmin < 1e-300 {
		return math.Inf(1)
	}
	return math.Sqrt(lmax / lmin)
}
