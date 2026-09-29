package solver

import (
	"fmt"
	"math"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// Preconditioner 是预处理器：求解 M*z = r。
// 预处理仅改变迭代路径（加速收敛），不得改变原问题 A*x=b 的解。
type Preconditioner interface {
	// Apply 求解 M*z = r，返回 z。
	Apply(r []float64) ([]float64, error)
	// Type 返回预处理器类型标识。
	Type() PrecondType
}

// SingularProblemError 表示检测到奇异或明显病态问题，无法继续可靠求解。
type SingularProblemError struct {
	Reason string
}

func (e *SingularProblemError) Error() string {
	return "solver: 奇异或明显病态的方程组: " + e.Reason
}

// NewPreconditioner 按配置为矩阵 A 构造预处理器。
// A 必须为方阵；对角元为零（Jacobi）或出现零主元（ILU(0)）时
// 返回 *SingularProblemError，与普通输入错误区分开。
func NewPreconditioner(kind PrecondType, A *sparse.CSR) (Preconditioner, error) {
	if kind == PrecondNone {
		return &identityPrecond{}, nil
	}
	n, m := A.Dims()
	if n != m {
		return nil, &sparse.FormatMismatchError{
			Operation: "NewPreconditioner", Got: fmt.Sprintf("%dx%d", n, m),
			Expect: "方阵", Detail: "预处理器要求系数矩阵为方阵",
		}
	}
	scale := matrixScale(A)
	switch kind {
	case PrecondJacobi:
		inv := make([]float64, n)
		for i := 0; i < n; i++ {
			aii := 0.0
			for k := A.RowPtr[i]; k < A.RowPtr[i+1]; k++ {
				if A.ColInd[k] == i {
					aii = A.Data[k]
					break
				}
			}
			if aii == 0 {
				return nil, &SingularProblemError{Reason: fmt.Sprintf(
					"Jacobi 预处理失败：第 %d 个对角元为零，矩阵奇异或不适合 Jacobi", i)}
			}
			// 对角元相对矩阵尺度极小，视为明显病态： Jacobi 会把残差放大到失真。
			if scale > 0 && math.Abs(aii) < 1e-14*scale {
				return nil, &SingularProblemError{Reason: fmt.Sprintf(
					"Jacobi 预处理失败：第 %d 个对角元 |a_ii|=%.3e 相对矩阵尺度 %.3e 过小，方程组明显病态",
					i, math.Abs(aii), scale)}
			}
			inv[i] = 1.0 / aii
		}
		return &Jacobi{invDiag: inv}, nil
	case PrecondILU0:
		return newILU0(A, scale)
	default:
		return nil, &sparse.FormatMismatchError{
			Operation: "NewPreconditioner", Got: fmt.Sprintf("PrecondType(%d)", kind),
			Expect: "None/Jacobi/ILU0", Detail: "未知预处理器类型",
		}
	}
}

// matrixScale 用最大绝对元素衡量矩阵的整体量级，供主元判据做相对比较。
func matrixScale(A *sparse.CSR) float64 {
	s := 0.0
	for _, v := range A.Data {
		if a := math.Abs(v); a > s {
			s = a
		}
	}
	return s
}

// identityPrecond 是 M=I 的空预处理器。
type identityPrecond struct{}

func (identityPrecond) Apply(r []float64) ([]float64, error) {
	z := make([]float64, len(r))
	copy(z, r)
	return z, nil
}
func (identityPrecond) Type() PrecondType { return PrecondNone }

// Jacobi 预处理器：M = diag(A)。
type Jacobi struct {
	invDiag []float64
}

func (p *Jacobi) Apply(r []float64) ([]float64, error) {
	if len(r) != len(p.invDiag) {
		return nil, &sparse.DimMismatchError{
			Operation: "Jacobi.Apply", Rows: len(p.invDiag), Cols: len(p.invDiag),
			VecLen: len(r), Detail: "r 长度必须等于矩阵阶数",
		}
	}
	z := make([]float64, len(r))
	for i := range r {
		z[i] = p.invDiag[i] * r[i]
	}
	return z, nil
}

func (p *Jacobi) Type() PrecondType { return PrecondJacobi }

// ILU0 预处理器：保持 A 稀疏结构的零填充不完全 LU 分解。
type ILU0 struct {
	n      int
	rowPtr []int
	colInd []int
	lu     []float64
}

func newILU0(A *sparse.CSR, scale float64) (*ILU0, error) {
	n := A.Rows
	lu := append([]float64(nil), A.Data...)
	rowPtr := A.RowPtr
	colInd := A.ColInd

	// colPos[k] 记录位置 k 处列索引（在同一行内）的快速定位；
	// ILU(0) 只允许更新 A 原有稀疏结构内的位置。
	for i := 1; i < n; i++ {
		rowStart, rowEnd := rowPtr[i], rowPtr[i+1]
		pos := make(map[int]int, rowEnd-rowStart)
		for k := rowStart; k < rowEnd; k++ {
			pos[colInd[k]] = k
		}
		// 先求行 i 对角元之前各 L 列的消去系数。
		for k := rowStart; k < rowEnd && colInd[k] < i; k++ {
			j := colInd[k]
			// 取 u_jj：第 j 行对角元。
			ujj := 0.0
			ujjK := -1
			for qk := rowPtr[j]; qk < rowPtr[j+1]; qk++ {
				if colInd[qk] == j {
					ujjK = qk
					break
				}
			}
			if ujjK < 0 {
				return nil, &SingularProblemError{Reason: fmt.Sprintf(
					"ILU(0) 失败：第 %d 行缺少对角结构，矩阵结构奇异", j)}
			}
			ujj = lu[ujjK]
			if ujj == 0 {
				return nil, &SingularProblemError{Reason: fmt.Sprintf(
					"ILU(0) 零主元：第 %d 步主元为 0，矩阵奇异", j)}
			}
			if scale > 0 && math.Abs(ujj) < 1e-14*scale {
				return nil, &SingularProblemError{Reason: fmt.Sprintf(
					"ILU(0) 主元过小：第 %d 步 |主元|=%.3e 相对矩阵尺度 %.3e，方程组明显病态",
					j, math.Abs(ujj), scale)}
			}
			lu[k] /= ujj
			lik := lu[k]
			// 更新行 i 中、第 j 行上三角部分与本行共有的结构位置。
			for qk := rowPtr[j]; qk < rowPtr[j+1]; qk++ {
				q := colInd[qk]
				if q <= j {
					continue // 只用 U 的第 j 行（列 > j）
				}
				if mk, ok := pos[q]; ok {
					lu[mk] -= lik * lu[qk]
				}
				// 不在 A 结构内的填充项按 ILU(0) 规则丢弃。
			}
		}
		// 检查本行主元。
		dk := -1
		for k := rowStart; k < rowEnd; k++ {
			if colInd[k] == i {
				dk = k
				break
			}
		}
		if dk < 0 {
			return nil, &SingularProblemError{Reason: fmt.Sprintf(
				"ILU(0) 失败：第 %d 行缺少对角结构，矩阵结构奇异", i)}
		}
		if lu[dk] == 0 {
			return nil, &SingularProblemError{Reason: fmt.Sprintf(
				"ILU(0) 零主元：第 %d 步主元为 0，矩阵奇异", i)}
		}
		if scale > 0 && math.Abs(lu[dk]) < 1e-14*scale {
			return nil, &SingularProblemError{Reason: fmt.Sprintf(
				"ILU(0) 主元过小：第 %d 步 |主元|=%.3e 相对矩阵尺度 %.3e，方程组明显病态",
				i, math.Abs(lu[dk]), scale)}
		}
	}
	return &ILU0{n: n, rowPtr: rowPtr, colInd: colInd, lu: lu}, nil
}

// Apply 求解 LU*z = r：先 L*y = r（前向，单位对角），再 U*z = y（后向）。
func (p *ILU0) Apply(r []float64) ([]float64, error) {
	if len(r) != p.n {
		return nil, &sparse.DimMismatchError{
			Operation: "ILU0.Apply", Rows: p.n, Cols: p.n, VecLen: len(r),
			Detail: "r 长度必须等于矩阵阶数",
		}
	}
	y := make([]float64, p.n)
	for i := 0; i < p.n; i++ {
		s := r[i]
		for k := p.rowPtr[i]; k < p.rowPtr[i+1] && p.colInd[k] < i; k++ {
			s -= p.lu[k] * y[p.colInd[k]]
		}
		y[i] = s // L 对角为 1
	}
	z := make([]float64, p.n)
	for i := p.n - 1; i >= 0; i-- {
		s := y[i]
		diagK := -1
		for k := p.rowPtr[i]; k < p.rowPtr[i+1]; k++ {
			j := p.colInd[k]
			if j > i {
				s -= p.lu[k] * z[j]
			} else if j == i {
				diagK = k
			}
		}
		if diagK < 0 || p.lu[diagK] == 0 {
			return nil, &SingularProblemError{Reason: "ILU(0) 回代时缺失主元"}
		}
		z[i] = s / p.lu[diagK]
	}
	return z, nil
}

func (p *ILU0) Type() PrecondType { return PrecondILU0 }
