package sparse

import "reflect"

// isNilMatrix 判断接口是否为空、或持有一个 nil 指针的矩阵值。
// 后者来自 (*COO)(nil) 之类的 typed-nil，不能直接与 nil 比较。
func isNilMatrix(A Matrix) bool {
	if A == nil {
		return true
	}
	v := reflect.ValueOf(A)
	switch v.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
		return v.IsNil()
	}
	return false
}

// IsNilMatrix 判断矩阵接口是否为空、或持有 nil 指针（typed-nil）。
func IsNilMatrix(A Matrix) bool { return isNilMatrix(A) }
