package sparse

// CSR 是行压缩格式（Compressed Sparse Row）稀疏矩阵。
// 第 i 行的非零元列索引为 ColInd[RowPtr[i]:RowPtr[i+1]]，
// 对应数值为 Data[RowPtr[i]:RowPtr[i+1]]。
type CSR struct {
	Rows   int
	Cols   int
	RowPtr []int // 长度 Rows+1
	ColInd []int // 每行按列索引升序
	Data   []float64
}

// validate 校验 CSR 内部结构的完整性，任何损坏都显式报错，
// 不静默当作零或越界容错。
func (a *CSR) validate(op string) error {
	if a == nil {
		return &FormatMismatchError{Operation: op, Got: "<nil>", Expect: "*CSR",
			Detail: "矩阵为空"}
	}
	if a.Rows < 0 || a.Cols < 0 {
		return &DimMismatchError{Operation: op, Rows: a.Rows, Cols: a.Cols,
			Detail: "矩阵维度必须非负"}
	}
	if len(a.RowPtr) != a.Rows+1 {
		return &FormatMismatchError{Operation: op, Got: "*CSR", Expect: "RowPtr 长度为 Rows+1",
			Detail: "行指针数组长度非法"}
	}
	if len(a.ColInd) != len(a.Data) {
		return &FormatMismatchError{Operation: op, Got: "*CSR", Expect: "len(ColInd)==len(Data)",
			Detail: "列索引与数值数组长度不一致"}
	}
	if a.RowPtr[0] != 0 || a.RowPtr[a.Rows] != len(a.Data) {
		return &FormatMismatchError{Operation: op, Got: "*CSR", Expect: "RowPtr 首项为0且末项为NNZ",
			Detail: "行指针端点非法"}
	}
	for i := 0; i < a.Rows; i++ {
		if a.RowPtr[i] > a.RowPtr[i+1] {
			return &FormatMismatchError{Operation: op, Got: "*CSR", Expect: "RowPtr 单调不减",
				Detail: "行指针不是单调不减的"}
		}
		for k := a.RowPtr[i]; k < a.RowPtr[i+1]; k++ {
			j := a.ColInd[k]
			if j < 0 || j >= a.Cols {
				return &IndexOutOfBoundsError{Operation: op, Index: j,
					Lower: 0, Upper: a.Cols, Role: "column"}
			}
			if k > a.RowPtr[i] && j <= a.ColInd[k-1] {
				return &FormatMismatchError{Operation: op, Got: "*CSR",
					Expect: "每行列索引严格升序", Detail: "存在乱序或重复列索引"}
			}
		}
	}
	return nil
}

// NewCSR 根据行指针、列索引与数值数组构造 CSR，并做完整性校验。
func NewCSR(rows, cols int, rowPtr, colInd []int, data []float64) (*CSR, error) {
	a := &CSR{
		Rows:   rows,
		Cols:   cols,
		RowPtr: append([]int(nil), rowPtr...),
		ColInd: append([]int(nil), colInd...),
		Data:   append([]float64(nil), data...),
	}
	if err := a.validate("NewCSR"); err != nil {
		return nil, err
	}
	return a, nil
}

// ToCOO 将 CSR 展开为三元组格式。
func (a *CSR) ToCOO() (*COO, error) {
	if err := a.validate("CSR.ToCOO"); err != nil {
		return nil, err
	}
	nnz := len(a.Data)
	ri := make([]int, nnz)
	ci := make([]int, nnz)
	vv := make([]float64, nnz)
	k := 0
	for i := 0; i < a.Rows; i++ {
		for ; k < a.RowPtr[i+1]; k++ {
			ri[k] = i
			ci[k] = a.ColInd[k]
			vv[k] = a.Data[k]
		}
	}
	return &COO{Rows: a.Rows, Cols: a.Cols, RowIndices: ri, ColIndices: ci, Data: vv}, nil
}

// ToCSC 将 CSR 转置压缩为 CSC 格式。
func (a *CSR) ToCSC() (*CSC, error) {
	if err := a.validate("CSR.ToCSC"); err != nil {
		return nil, err
	}
	nnz := len(a.Data)
	colPtr := make([]int, a.Cols+1)
	rowInd := make([]int, nnz)
	data := make([]float64, nnz)
	for k := 0; k < nnz; k++ {
		colPtr[a.ColInd[k]+1]++
	}
	for j := 1; j <= a.Cols; j++ {
		colPtr[j] += colPtr[j-1]
	}
	// dest 跟踪每一列当前写入位置。
	dest := append([]int(nil), colPtr[:a.Cols]...)
	for i := 0; i < a.Rows; i++ {
		for k := a.RowPtr[i]; k < a.RowPtr[i+1]; k++ {
			j := a.ColInd[k]
			d := dest[j]
			dest[j]++
			rowInd[d] = i
			data[d] = a.Data[k]
		}
	}
	csc := &CSC{Rows: a.Rows, Cols: a.Cols, ColPtr: colPtr, RowInd: rowInd, Data: data}
	if err := csc.validate("CSR.ToCSC"); err != nil {
		return nil, err
	}
	return csc, nil
}

// MatVec 计算 y = A*x（实现 Matrix 接口）。
// x 长度与 A 列数不一致时返回 DimMismatchError。
func (a *CSR) MatVec(x []float64) ([]float64, error) {
	if err := a.validate("CSR.MatVec"); err != nil {
		return nil, err
	}
	if len(x) != a.Cols {
		return nil, &DimMismatchError{
			Operation: "CSR.MatVec", Rows: a.Rows, Cols: a.Cols, VecLen: len(x),
			Detail: "x 长度必须等于矩阵列数",
		}
	}
	y := make([]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		s := 0.0
		for k := a.RowPtr[i]; k < a.RowPtr[i+1]; k++ {
			s += a.Data[k] * x[a.ColInd[k]]
		}
		y[i] = s
	}
	return y, nil
}

// Dims 返回矩阵行数与列数。
func (a *CSR) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ 返回显式存储的非零元个数。
func (a *CSR) NNZ() int { return len(a.Data) }

// ToCSR 返回自身的校验后深拷贝（满足 Convertible 接口）。
func (a *CSR) ToCSR() (*CSR, error) {
	if err := a.validate("CSR.ToCSR"); err != nil {
		return nil, err
	}
	return &CSR{
		Rows:   a.Rows,
		Cols:   a.Cols,
		RowPtr: append([]int(nil), a.RowPtr...),
		ColInd: append([]int(nil), a.ColInd...),
		Data:   append([]float64(nil), a.Data...),
	}, nil
}
