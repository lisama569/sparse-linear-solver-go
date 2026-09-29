// Package sparse 提供稀疏矩阵的多种压缩存储格式及基本运算。
package sparse

import "fmt"

// DimMismatchError 表示矩阵维度与向量长度等不一致，无法完成运算。
type DimMismatchError struct {
	Operation string // 触发错误的运算名称
	Rows      int    // 矩阵行数（或第一维长度）
	Cols      int    // 矩阵列数（或第二维长度）
	VecLen    int    // 参与运算的向量长度（为 0 表示不涉及）
	Detail    string // 补充说明
}

func (e *DimMismatchError) Error() string {
	if e.VecLen > 0 {
		return fmt.Sprintf("sparse: %s: 维度不匹配: 矩阵 %dx%d, 向量长度 %d (%s)",
			e.Operation, e.Rows, e.Cols, e.VecLen, e.Detail)
	}
	return fmt.Sprintf("sparse: %s: 维度不匹配: %dx%d (%s)",
		e.Operation, e.Rows, e.Cols, e.Detail)
}

// IndexOutOfBoundsError 表示三元组或指针中出现了超出矩阵维度的索引。
// 该错误与"结构零"严格区分：越界输入绝不被静默当作零。
type IndexOutOfBoundsError struct {
	Operation string // 触发错误的运算名称
	Index     int    // 越界的索引值
	Lower     int    // 合法下界（通常为 0）
	Upper     int    // 合法上界（不含）
	Role      string // 索引角色，如 "row"、"column"
}

func (e *IndexOutOfBoundsError) Error() string {
	return fmt.Sprintf("sparse: %s: 索引越界: %s=%d 不在 [%d,%d) 范围内",
		e.Operation, e.Role, e.Index, e.Lower, e.Upper)
}

// FormatMismatchError 表示输入矩阵的存储格式不被当前运算接受
// （例如只接受方阵或不支持的压缩格式）。
type FormatMismatchError struct {
	Operation string // 触发错误的运算名称
	Got       string // 实际收到的格式名称
	Expect    string // 期望的格式（或约束）
	Detail    string // 补充说明
}

func (e *FormatMismatchError) Error() string {
	return fmt.Sprintf("sparse: %s: 格式不匹配: 收到 %s, 期望 %s (%s)",
		e.Operation, e.Got, e.Expect, e.Detail)
}
