package sparse

import "errors"

// 可区分的错误类别：格式不支持、维度不匹配、索引越界、数值失效。
var (
	ErrDimensionMismatch = errors.New("sparse: dimension mismatch")
	ErrIndexOutOfRange   = errors.New("sparse: index out of range")
	ErrZeroPivot         = errors.New("sparse: zero pivot encountered")
	ErrNotSymmetric      = errors.New("sparse: matrix must be symmetric positive definite for this solver")
)
