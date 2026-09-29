package sparse

// CSC 是列压缩格式（Compressed Sparse Column）稀疏矩阵。
// 第 j 列的非零元行索引为 RowInd[ColPtr[j]:ColPtr[j+1]]，
// 对应数值为 Data[ColPtr[j]:ColPtr[j+1]]。
type CSC struct {
	Rows   int
	Cols   int
	ColPtr []int // 长度 Cols+1
	RowInd []int // 每列按行索引升序
	Data   []float64
}

// validate 校验 CSC 内部结构的完整性。
func (a *CSC) validate(op string) error {
	if a == nil {
		return &FormatMismatchError{Operation: op, Got: "<nil>", Expect: "*CSC",
			Detail: "矩阵为空"}
	}
	if a.Rows < 0 || a.Cols < 0 {
		return &DimMismatchError{Operation: op, Rows: a.Rows, Cols: a.Cols,
			Detail: "矩阵维度必须非负"}
	}
	if len(a.ColPtr) != a.Cols+1 {
		return &FormatMismatchError{Operation: op, Got: "*CSC", Expect: "ColPtr 长度为 Cols+1",
			Detail: "列指针数组长度非法"}
	}
	if len(a.RowInd) != len(a.Data) {
		return &FormatMismatchError{Operation: op, Got: "*CSC", Expect: "len(RowInd)==len(Data)",
			Detail: "行索引与数值数组长度不一致"}
	}
	if a.ColPtr[0] != 0 || a.ColPtr[a.Cols] != len(a.Data) {
		return &FormatMismatchError{Operation: op, Got: "*CSC", Expect: "ColPtr 首项为0且末项为NNZ",
			Detail: "列指针端点非法"}
	}
	for j := 0; j < a.Cols; j++ {
		if a.ColPtr[j] > a.ColPtr[j+1] {
			return &FormatMismatchError{Operation: op, Got: "*CSC", Expect: "ColPtr 单调不减",
				Detail: "列指针不是单调不减的"}
		}
		for k := a.ColPtr[j]; k < a.ColPtr[j+1]; k++ {
			i := a.RowInd[k]
			if i < 0 || i >= a.Rows {
				return &IndexOutOfBoundsError{Operation: op, Index: i,
					Lower: 0, Upper: a.Rows, Role: "row"}
			}
			if k > a.ColPtr[j] && i <= a.RowInd[k-1] {
				return &FormatMismatchError{Operation: op, Got: "*CSC",
					Expect: "每列行索引严格升序", Detail: "存在乱序或重复行索引"}
			}
		}
	}
	return nil
}

// NewCSC 根据列指针、行索引与数值数组构造 CSC，并做完整性校验。
func NewCSC(rows, cols int, colPtr, rowInd []int, data []float64) (*CSC, error) {
	a := &CSC{
		Rows:   rows,
		Cols:   cols,
		ColPtr: append([]int(nil), colPtr...),
		RowInd: append([]int(nil), rowInd...),
		Data:   append([]float64(nil), data...),
	}
	if err := a.validate("NewCSC"); err != nil {
		return nil, err
	}
	return a, nil
}

// ToCOO 将 CSC 展开为三元组格式。
func (a *CSC) ToCOO() (*COO, error) {
	if err := a.validate("CSC.ToCOO"); err != nil {
		return nil, err
	}
	nnz := len(a.Data)
	ri := make([]int, nnz)
	ci := make([]int, nnz)
	vv := make([]float64, nnz)
	k := 0
	for j := 0; j < a.Cols; j++ {
		for ; k < a.ColPtr[j+1]; k++ {
			ri[k] = a.RowInd[k]
			ci[k] = j
			vv[k] = a.Data[k]
		}
	}
	return &COO{Rows: a.Rows, Cols: a.Cols, RowIndices: ri, ColIndices: ci, Data: vv}, nil
}

// ToCSR 将 CSC 转置压缩为 CSR 格式。
func (a *CSC) ToCSR() (*CSR, error) {
	if err := a.validate("CSC.ToCSR"); err != nil {
		return nil, err
	}
	nnz := len(a.Data)
	rowPtr := make([]int, a.Rows+1)
	colInd := make([]int, nnz)
	data := make([]float64, nnz)
	for k := 0; k < nnz; k++ {
		rowPtr[a.RowInd[k]+1]++
	}
	for i := 1; i <= a.Rows; i++ {
		rowPtr[i] += rowPtr[i-1]
	}
	dest := append([]int(nil), rowPtr[:a.Rows]...)
	for j := 0; j < a.Cols; j++ {
		for k := a.ColPtr[j]; k < a.ColPtr[j+1]; k++ {
			i := a.RowInd[k]
			d := dest[i]
			dest[i]++
			colInd[d] = j
			data[d] = a.Data[k]
		}
	}
	csr := &CSR{Rows: a.Rows, Cols: a.Cols, RowPtr: rowPtr, ColInd: colInd, Data: data}
	if err := csr.validate("CSC.ToCSR"); err != nil {
		return nil, err
	}
	return csr, nil
}

// MatVec 计算 y = A*x（实现 Matrix 接口）。
// x 长度与 A 列数不一致时返回 DimMismatchError。
func (a *CSC) MatVec(x []float64) ([]float64, error) {
	if err := a.validate("CSC.MatVec"); err != nil {
		return nil, err
	}
	if len(x) != a.Cols {
		return nil, &DimMismatchError{
			Operation: "CSC.MatVec", Rows: a.Rows, Cols: a.Cols, VecLen: len(x),
			Detail: "x 长度必须等于矩阵列数",
		}
	}
	y := make([]float64, a.Rows)
	for j := 0; j < a.Cols; j++ {
		xj := x[j]
		for k := a.ColPtr[j]; k < a.ColPtr[j+1]; k++ {
			y[a.RowInd[k]] += a.Data[k] * xj
		}
	}
	return y, nil
}

// Dims 返回矩阵行数与列数。
func (a *CSC) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ 返回显式存储的非零元个数。
func (a *CSC) NNZ() int { return len(a.Data) }

// ToCSC 返回自身的校验后深拷贝（满足 Convertible 接口）。
func (a *CSC) ToCSC() (*CSC, error) {
	if err := a.validate("CSC.ToCSC"); err != nil {
		return nil, err
	}
	return &CSC{
		Rows:   a.Rows,
		Cols:   a.Cols,
		ColPtr: append([]int(nil), a.ColPtr...),
		RowInd: append([]int(nil), a.RowInd...),
		Data:   append([]float64(nil), a.Data...),
	}, nil
}
