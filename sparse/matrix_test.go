package sparse

import (
	"errors"
	"math"
	"testing"
)

// 同一逻辑矩阵的三种格式，MatVec 结果必须一致。
func TestFormatConversionsAndMatVec(t *testing.T) {
	entries := []Entry{
		{0, 0, 4}, {0, 2, 1},
		{1, 1, 3}, {1, 2, -1},
		{2, 0, 1}, {2, 2, 2},
		{0, 0, 0.5}, // 重复坐标，应累加为 4.5
	}
	coo, err := NewCOO(3, 3, entries)
	if err != nil {
		t.Fatalf("NewCOO: %v", err)
	}
	csr := coo.ToCSR()
	csc := coo.ToCSC()

	if csr.NNZ() != 6 {
		t.Fatalf("CSR NNZ = %d, want 6 (duplicates merged)", csr.NNZ())
	}
	// 往返转换后应保持一致
	if got := csr.ToCOO().ToCSR().NNZ(); got != 6 {
		t.Fatalf("CSR->COO->CSR NNZ = %d, want 6", got)
	}
	if got := csc.ToCSR().NNZ(); got != 6 {
		t.Fatalf("CSC->CSR NNZ = %d, want 6", got)
	}
	if got := csr.ToCSC().NNZ(); got != 6 {
		t.Fatalf("CSR->CSC NNZ = %d, want 6", got)
	}

	x := []float64{1, 2, 3}
	want := []float64{4.5*1 + 1*3, 3*2 - 1*3, 1*1 + 2*3}
	for name, m := range map[string]Matrix{"COO": coo, "CSR": csr, "CSC": csc} {
		y, err := m.MatVec(x)
		if err != nil {
			t.Fatalf("%s MatVec: %v", name, err)
		}
		for i := range want {
			if math.Abs(y[i]-want[i]) > 1e-12 {
				t.Errorf("%s MatVec y[%d] = %v, want %v", name, i, y[i], want[i])
			}
		}
	}
}

func TestMatVecDimensionMismatch(t *testing.T) {
	coo, _ := NewCOO(2, 3, []Entry{{0, 0, 1}})
	for name, m := range map[string]Matrix{"COO": coo, "CSR": coo.ToCSR(), "CSC": coo.ToCSC()} {
		if _, err := m.MatVec([]float64{1, 2}); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s MatVec err = %v, want ErrDimensionMismatch", name, err)
		}
	}
}

func TestNewCOOIndexOutOfRange(t *testing.T) {
	if _, err := NewCOO(2, 2, []Entry{{2, 0, 1}}); !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("row overflow err = %v, want ErrIndexOutOfRange", err)
	}
	if _, err := NewCOO(2, 2, []Entry{{0, -1, 1}}); !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("negative col err = %v, want ErrIndexOutOfRange", err)
	}
}
