package sparse

import "sort"

// COO 是坐标格式（Coordinate / Triplet）稀疏矩阵。
// 第 k 个非零元为 Data[k]，位置 (RowIndices[k], ColIndices[k])。
// 允许重复坐标，按代数累加语义处理；允许乱序。
type COO struct {
	Rows       int
	Cols       int
	RowIndices []int
	ColIndices []int
	Data       []float64
}

// NewCOO 创建 rows x cols 的 COO 矩阵并追加三元组。
// 维度非法或索引越界时返回对应错误，绝不静默丢弃或当零处理。
func NewCOO(rows, cols int, rowIdx, colIdx []int, data []float64) (*COO, error) {
	if rows < 0 || cols < 0 {
		return nil, &DimMismatchError{
			Operation: "NewCOO", Rows: rows, Cols: cols,
			Detail: "矩阵维度必须非负",
		}
	}
	if len(rowIdx) != len(colIdx) || len(rowIdx) != len(data) {
		return nil, &DimMismatchError{
			Operation: "NewCOO", Rows: rows, Cols: cols, VecLen: len(data),
			Detail: "rowIdx/colIdx/data 三个数组长度必须一致",
		}
	}
	for k, i := range rowIdx {
		if i < 0 || i >= rows {
			return nil, &IndexOutOfBoundsError{
				Operation: "NewCOO", Index: i, Lower: 0, Upper: rows,
				Role: "row",
			}
		}
		j := colIdx[k]
		if j < 0 || j >= cols {
			return nil, &IndexOutOfBoundsError{
				Operation: "NewCOO", Index: j, Lower: 0, Upper: cols,
				Role: "column",
			}
		}
	}
	// 拷贝入参，避免调用方后续修改破坏矩阵一致性。
	ri := append([]int(nil), rowIdx...)
	ci := append([]int(nil), colIdx...)
	vv := append([]float64(nil), data...)
	return &COO{Rows: rows, Cols: cols, RowIndices: ri, ColIndices: ci, Data: vv}, nil
}

// triplet 是排序用的单个条目。
type triplet struct {
	row, col int
	val      float64
}

// sortedTriplets 校验自身维度并返回按 (行,列) 升序排列的条目，
// 重复坐标已累加，累加后恰为 0 的条目被剔除。
func (a *COO) sortedTriplets(op string) ([]triplet, error) {
	if a == nil || a.Rows < 0 || a.Cols < 0 {
		return nil, &DimMismatchError{Operation: op, Detail: "COO 矩阵维度非法"}
	}
	if len(a.RowIndices) != len(a.ColIndices) || len(a.RowIndices) != len(a.Data) {
		return nil, &DimMismatchError{
			Operation: op, Rows: a.Rows, Cols: a.Cols, VecLen: len(a.Data),
			Detail: "rowIdx/colIdx/data 三个数组长度必须一致",
		}
	}
	ts := make([]triplet, len(a.Data))
	for k := range a.Data {
		i, j := a.RowIndices[k], a.ColIndices[k]
		if i < 0 || i >= a.Rows {
			return nil, &IndexOutOfBoundsError{
				Operation: op, Index: i, Lower: 0, Upper: a.Rows, Role: "row",
			}
		}
		if j < 0 || j >= a.Cols {
			return nil, &IndexOutOfBoundsError{
				Operation: op, Index: j, Lower: 0, Upper: a.Cols, Role: "column",
			}
		}
		ts[k] = triplet{i, j, a.Data[k]}
	}
	sort.Slice(ts, func(p, q int) bool {
		if ts[p].row != ts[q].row {
			return ts[p].row < ts[q].row
		}
		return ts[p].col < ts[q].col
	})
	// 累加重复坐标。
	merged := ts[:0]
	for _, t := range ts {
		if n := len(merged); n > 0 && merged[n-1].row == t.row && merged[n-1].col == t.col {
			merged[n-1].val += t.val
		} else {
			merged = append(merged, t)
		}
	}
	// 剔除累加后恰为零的结构项（零值不影响语义）。
	out := merged[:0]
	for _, t := range merged {
		if t.val != 0 {
			out = append(out, t)
		}
	}
	return out, nil
}

// ToCSR 将 COO 转换为 CSR（行压缩）格式，重复坐标自动累加。
func (a *COO) ToCSR() (*CSR, error) {
	ts, err := a.sortedTriplets("COO.ToCSR")
	if err != nil {
		return nil, err
	}
	rowPtr := make([]int, a.Rows+1)
	colInd := make([]int, 0, len(ts))
	data := make([]float64, 0, len(ts))
	for _, t := range ts {
		rowPtr[t.row]++
		colInd = append(colInd, t.col)
		data = append(data, t.val)
	}
	// 计数转偏移。
	sum := 0
	for i := 0; i < a.Rows; i++ {
		c := rowPtr[i]
		rowPtr[i] = sum
		sum += c
	}
	rowPtr[a.Rows] = sum
	return &CSR{Rows: a.Rows, Cols: a.Cols, RowPtr: rowPtr, ColInd: colInd, Data: data}, nil
}

// ToCSC 将 COO 转换为 CSC（列压缩）格式，重复坐标自动累加。
func (a *COO) ToCSC() (*CSC, error) {
	ts, err := a.sortedTriplets("COO.ToCSC")
	if err != nil {
		return nil, err
	}
	// sortedTriplets 按 (row,col) 排序；CSC 需要按 (col,row)，重排一次。
	sort.SliceStable(ts, func(p, q int) bool {
		if ts[p].col != ts[q].col {
			return ts[p].col < ts[q].col
		}
		return ts[p].row < ts[q].row
	})
	colPtr := make([]int, a.Cols+1)
	rowInd := make([]int, 0, len(ts))
	data := make([]float64, 0, len(ts))
	for _, t := range ts {
		colPtr[t.col]++
		rowInd = append(rowInd, t.row)
		data = append(data, t.val)
	}
	sum := 0
	for j := 0; j < a.Cols; j++ {
		c := colPtr[j]
		colPtr[j] = sum
		sum += c
	}
	colPtr[a.Cols] = sum
	return &CSC{Rows: a.Rows, Cols: a.Cols, ColPtr: colPtr, RowInd: rowInd, Data: data}, nil
}

// Dims 返回矩阵行数与列数。
func (a *COO) Dims() (int, int) { return a.Rows, a.Cols }

// NNZ 返回存储的（累加前）条目数。
func (a *COO) NNZ() int { return len(a.Data) }

// ToCOO 返回经校验、重复坐标累加并重排后的深拷贝（满足 Convertible 接口）。
func (a *COO) ToCOO() (*COO, error) {
	ts, err := a.sortedTriplets("COO.ToCOO")
	if err != nil {
		return nil, err
	}
	out := &COO{
		Rows:       a.Rows,
		Cols:       a.Cols,
		RowIndices: make([]int, len(ts)),
		ColIndices: make([]int, len(ts)),
		Data:       make([]float64, len(ts)),
	}
	for k, t := range ts {
		out.RowIndices[k] = t.row
		out.ColIndices[k] = t.col
		out.Data[k] = t.val
	}
	return out, nil
}

// MatVec 计算 y = A*x（实现 Matrix 接口）。
// 先转换为 CSR 再执行；重复坐标按累加语义处理。
func (a *COO) MatVec(x []float64) ([]float64, error) {
	if a == nil {
		return nil, &FormatMismatchError{Operation: "COO.MatVec", Got: "<nil>",
			Expect: "*COO", Detail: "矩阵为空"}
	}
	if len(x) != a.Cols {
		return nil, &DimMismatchError{
			Operation: "COO.MatVec", Rows: a.Rows, Cols: a.Cols, VecLen: len(x),
			Detail: "x 长度必须等于矩阵列数",
		}
	}
	csr, err := a.ToCSR()
	if err != nil {
		return nil, err
	}
	return csr.MatVec(x)
}
