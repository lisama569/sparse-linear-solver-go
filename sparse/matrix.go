package sparse

import "math"

// Matrix 是所有稀疏存储格式共同实现的只读矩阵接口。
type Matrix interface {
	// Dims 返回矩阵行数与列数。
	Dims() (int, int)
	// NNZ 返回显式存储的非零元个数。
	NNZ() int
	// MatVec 计算 y = A*x。
	MatVec(x []float64) ([]float64, error)
}

// Convertible 表示可在三种压缩存储格式之间互相转换的稀疏矩阵。
type Convertible interface {
	ToCOO() (*COO, error)
	ToCSR() (*CSR, error)
	ToCSC() (*CSC, error)
}

// MatVec 是统一的稀疏矩阵-向量乘入口 y = A*x。
// A 与 x 维度不匹配时返回 *DimMismatchError；
// 矩阵内部结构损坏时可能返回 *IndexOutOfBoundsError 或 *FormatMismatchError。
func MatVec(A Matrix, x []float64) ([]float64, error) {
	if isNilMatrix(A) {
		return nil, &FormatMismatchError{
			Operation: "MatVec", Got: "<nil>", Expect: "非空稀疏矩阵",
			Detail: "矩阵不能为空",
		}
	}
	rows, cols := A.Dims()
	if len(x) != cols {
		return nil, &DimMismatchError{
			Operation: "MatVec", Rows: rows, Cols: cols, VecLen: len(x),
			Detail: "x 长度必须等于矩阵列数",
		}
	}
	return A.MatVec(x)
}

// ResidualNorm 计算残差 r = b - A*x 的 2-范数。
// b 长度与 A 行数不一致时返回 *DimMismatchError。
func ResidualNorm(A Matrix, x, b []float64) (float64, error) {
	rows, cols := A.Dims()
	if len(b) != rows {
		return 0, &DimMismatchError{
			Operation: "ResidualNorm", Rows: rows, Cols: cols, VecLen: len(b),
			Detail: "b 长度必须等于矩阵行数",
		}
	}
	if len(x) != cols {
		return 0, &DimMismatchError{
			Operation: "ResidualNorm", Rows: rows, Cols: cols, VecLen: len(x),
			Detail: "x 长度必须等于矩阵列数",
		}
	}
	ax, err := MatVec(A, x)
	if err != nil {
		return 0, err
	}
	s := 0.0
	for i := range b {
		d := b[i] - ax[i]
		s += d * d
	}
	return math.Sqrt(s), nil
}

// FrobeniusNormSq 返回矩阵显式存储元素的平方和（Frobenius 范数平方）。
// 对不支持格式转换的自定义 Matrix 实现返回 0。
func FrobeniusNormSq(A Matrix) float64 {
	c, ok := A.(Convertible)
	if !ok {
		return 0
	}
	coo, err := c.ToCOO()
	if err != nil {
		return 0
	}
	s := 0.0
	for _, v := range coo.Data {
		s += v * v
	}
	return s
}
