package sparse

import "sort"

// Matrix 是所有稀疏存储格式都要实现的统一接口。
type Matrix interface {
	// Dims 返回矩阵的行数与列数。
	Dims() (rows, cols int)
	// NNZ 返回显式存储的非零元个数。
	NNZ() int
	// MatVec 计算 y = A*x，维度不匹配时返回 ErrDimensionMismatch。
	MatVec(x []float64) ([]float64, error)
}

// Entry 是坐标格式下的一个非零元。
type Entry struct {
	Row, Col int
	Value    float64
}

// COO 坐标格式。
type COO struct {
	Rows, Cols int
	Entries    []Entry
}

// CSR 压缩行存储格式。
type CSR struct {
	Rows, Cols int
	RowPtr     []int // 长度 Rows+1
	ColIdx     []int // 长度 NNZ
	Values     []float64
}

// CSC 压缩列存储格式。
type CSC struct {
	Rows, Cols int
	ColPtr     []int // 长度 Cols+1
	RowIdx     []int // 长度 NNZ
	Values     []float64
}

// NewCOO 构造坐标格式矩阵，索引越界返回 ErrIndexOutOfRange。
func NewCOO(rows, cols int, entries []Entry) (*COO, error) {
	if rows < 0 || cols < 0 {
		return nil, ErrDimensionMismatch
	}
	for _, e := range entries {
		if e.Row < 0 || e.Row >= rows || e.Col < 0 || e.Col >= cols {
			return nil, ErrIndexOutOfRange
		}
	}
	cp := make([]Entry, len(entries))
	copy(cp, entries)
	return &COO{Rows: rows, Cols: cols, Entries: cp}, nil
}

func (m *COO) Dims() (int, int) { return m.Rows, m.Cols }
func (m *COO) NNZ() int         { return len(m.Entries) }

func (m *COO) MatVec(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, ErrDimensionMismatch
	}
	y := make([]float64, m.Rows)
	for _, e := range m.Entries {
		y[e.Row] += e.Value * x[e.Col]
	}
	return y, nil
}

// ToCSR 将 COO 转为 CSR（相同坐标的重复项会累加）。
func (m *COO) ToCSR() *CSR {
	entries := make([]Entry, len(m.Entries))
	copy(entries, m.Entries)
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].Row != entries[j].Row {
			return entries[i].Row < entries[j].Row
		}
		return entries[i].Col < entries[j].Col
	})
	csr := &CSR{Rows: m.Rows, Cols: m.Cols, RowPtr: make([]int, m.Rows+1)}
	lastRow, lastCol := -1, -1
	for _, e := range entries {
		if e.Row == lastRow && e.Col == lastCol {
			csr.Values[len(csr.Values)-1] += e.Value // 累加同坐标重复项
			continue
		}
		lastRow, lastCol = e.Row, e.Col
		csr.ColIdx = append(csr.ColIdx, e.Col)
		csr.Values = append(csr.Values, e.Value)
		csr.RowPtr[e.Row+1]++
	}
	for i := 1; i <= m.Rows; i++ {
		csr.RowPtr[i] += csr.RowPtr[i-1]
	}
	return csr
}

// ToCSC 将 COO 转为 CSC（相同坐标的重复项会累加）。
func (m *COO) ToCSC() *CSC {
	return m.ToCSR().transposeToCSC()
}

func (m *CSR) Dims() (int, int) { return m.Rows, m.Cols }
func (m *CSR) NNZ() int         { return len(m.Values) }

func (m *CSR) MatVec(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, ErrDimensionMismatch
	}
	y := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		var sum float64
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			sum += m.Values[k] * x[m.ColIdx[k]]
		}
		y[i] = sum
	}
	return y, nil
}

// ToCOO 将 CSR 展开为 COO。
func (m *CSR) ToCOO() *COO {
	entries := make([]Entry, 0, len(m.Values))
	for i := 0; i < m.Rows; i++ {
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			entries = append(entries, Entry{Row: i, Col: m.ColIdx[k], Value: m.Values[k]})
		}
	}
	return &COO{Rows: m.Rows, Cols: m.Cols, Entries: entries}
}

// ToCSC 将 CSR 转为 CSC。
func (m *CSR) ToCSC() *CSC { return m.transposeToCSC() }

// transposeToCSC 把 CSR(A) 视为 CSR(A^T) 的构造流程，输出 CSC(A)。
func (m *CSR) transposeToCSC() *CSC {
	csc := &CSC{Rows: m.Rows, Cols: m.Cols, ColPtr: make([]int, m.Cols+1)}
	nnz := len(m.Values)
	csc.RowIdx = make([]int, nnz)
	csc.Values = make([]float64, nnz)
	for _, c := range m.ColIdx {
		csc.ColPtr[c+1]++
	}
	for j := 1; j <= m.Cols; j++ {
		csc.ColPtr[j] += csc.ColPtr[j-1]
	}
	next := make([]int, m.Cols+1)
	copy(next, csc.ColPtr)
	for i := 0; i < m.Rows; i++ {
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			c := m.ColIdx[k]
			p := next[c]
			csc.RowIdx[p] = i
			csc.Values[p] = m.Values[k]
			next[c]++
		}
	}
	return csc
}

func (m *CSC) Dims() (int, int) { return m.Rows, m.Cols }
func (m *CSC) NNZ() int         { return len(m.Values) }

func (m *CSC) MatVec(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, ErrDimensionMismatch
	}
	y := make([]float64, m.Rows)
	for j := 0; j < m.Cols; j++ {
		xj := x[j]
		if xj == 0 {
			continue
		}
		for k := m.ColPtr[j]; k < m.ColPtr[j+1]; k++ {
			y[m.RowIdx[k]] += m.Values[k] * xj
		}
	}
	return y, nil
}

// ToCOO 将 CSC 展开为 COO。
func (m *CSC) ToCOO() *COO {
	entries := make([]Entry, 0, len(m.Values))
	for j := 0; j < m.Cols; j++ {
		for k := m.ColPtr[j]; k < m.ColPtr[j+1]; k++ {
			entries = append(entries, Entry{Row: m.RowIdx[k], Col: j, Value: m.Values[k]})
		}
	}
	return &COO{Rows: m.Rows, Cols: m.Cols, Entries: entries}
}

// ToCSR 将 CSC 转为 CSR。
func (m *CSC) ToCSR() *CSR {
	// CSC(A) 当作 CSR(A^T) 转置，得到 CSR(A)
	t := &CSR{Rows: m.Cols, Cols: m.Rows, RowPtr: m.ColPtr, ColIdx: m.RowIdx, Values: m.Values}
	return t.transposeToCSC().toCSRFromTranspose(m.Rows, m.Cols)
}

// toCSRFromTranspose 在 transposeToCSC 结果上把 CSC 字段重命名回 CSR 语义。
func (c *CSC) toCSRFromTranspose(rows, cols int) *CSR {
	return &CSR{Rows: rows, Cols: cols, RowPtr: c.ColPtr, ColIdx: c.RowIdx, Values: c.Values}
}
