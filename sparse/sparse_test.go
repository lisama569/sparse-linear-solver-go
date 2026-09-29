package sparse

import (
	"errors"
	"math"
	"testing"
)

// buildDemoCOO 构造一个 3x3 演示矩阵（含重复坐标与乱序）：
//
//	[ 4  1  0 ]
//	[ 1  3  2 ]
//	[ 0  2  5 ]
func buildDemoCOO(t *testing.T) *COO {
	t.Helper()
	// (0,1)=1 被拆成两个 0.5 验证重复坐标累加。
	a, err := NewCOO(3, 3,
		[]int{2, 0, 1, 0, 2, 1, 0, 1},
		[]int{2, 0, 0, 1, 1, 1, 1, 2},
		[]float64{5, 4, 1, 0.5, 2, 3, 0.5, 2})
	if err != nil {
		t.Fatalf("NewCOO 失败: %v", err)
	}
	return a
}

func vecNear(a, b []float64, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func TestConversionsRoundTrip(t *testing.T) {
	coo := buildDemoCOO(t)

	csr, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("COO->CSR: %v", err)
	}
	wantPtr := []int{0, 2, 5, 7}
	if len(csr.RowPtr) != 4 {
		t.Fatalf("RowPtr 长度错误: %v", csr.RowPtr)
	}
	for i := range wantPtr {
		if csr.RowPtr[i] != wantPtr[i] {
			t.Fatalf("RowPtr=%v, 期望 %v", csr.RowPtr, wantPtr)
		}
	}
	if csr.NNZ() != 7 {
		t.Fatalf("CSR NNZ=%d, 期望 7（重复坐标累加后）", csr.NNZ())
	}

	csc, err := csr.ToCSC()
	if err != nil {
		t.Fatalf("CSR->CSC: %v", err)
	}
	wantColPtr := []int{0, 2, 5, 7}
	for i := range wantColPtr {
		if csc.ColPtr[i] != wantColPtr[i] {
			t.Fatalf("ColPtr=%v, 期望 %v", csc.ColPtr, wantColPtr)
		}
	}

	// 回转 COO 并对比稠密结果。
	back, err := csc.ToCOO()
	if err != nil {
		t.Fatalf("CSC->COO: %v", err)
	}
	csr2, err := back.ToCSR()
	if err != nil {
		t.Fatalf("COO->CSR(回转): %v", err)
	}
	for i := range csr.RowPtr {
		if csr.RowPtr[i] != csr2.RowPtr[i] {
			t.Fatalf("回转后 RowPtr 不一致: %v vs %v", csr2.RowPtr, csr.RowPtr)
		}
	}

	x := []float64{1, 2, 3}
	wantY := []float64{6, 13, 19}
	for name, M := range map[string]Matrix{"COO": mustCSR(t, coo), "CSR": csr, "CSC": csc} {
		y, err := MatVec(M, x)
		if err != nil {
			t.Fatalf("%s MatVec: %v", name, err)
		}
		if !vecNear(y, wantY, 1e-12) {
			t.Fatalf("%s MatVec=%v, 期望 %v", name, y, wantY)
		}
	}
}

func mustCSR(t *testing.T, c *COO) *CSR {
	t.Helper()
	r, err := c.ToCSR()
	if err != nil {
		t.Fatalf("ToCSR: %v", err)
	}
	return r
}

func TestMatVecDimMismatch(t *testing.T) {
	csr, _ := buildDemoCOO(t).ToCSR()
	_, err := MatVec(csr, []float64{1, 2}) // 长度 2 != 列数 3
	var dm *DimMismatchError
	if !errors.As(err, &dm) {
		t.Fatalf("期望 DimMismatchError, 实际 %T: %v", err, err)
	}
	t.Logf("维度不匹配被正确识别: %v", err)
}

func TestIndexOutOfBounds(t *testing.T) {
	_, err := NewCOO(2, 2, []int{0, 2}, []int{0, 1}, []float64{1, 1})
	var ib *IndexOutOfBoundsError
	if !errors.As(err, &ib) {
		t.Fatalf("期望 IndexOutOfBoundsError, 实际 %T: %v", err, err)
	}
	if ib.Role != "row" || ib.Upper != 2 {
		t.Fatalf("越界信息不精确: %+v", ib)
	}
	t.Logf("索引越界被正确识别: %v", err)

	_, err = NewCOO(2, 2, []int{0}, []int{-1}, []float64{1})
	if !errors.As(err, &ib) {
		t.Fatalf("负索引期望 IndexOutOfBoundsError, 实际 %v", err)
	}
}

func TestCSRStructureErrors(t *testing.T) {
	// 行指针端点损坏。
	_, err := NewCSR(2, 2, []int{1, 2, 3}, []int{0, 1, 0}, []float64{1, 2, 3})
	var fm *FormatMismatchError
	if !errors.As(err, &fm) {
		t.Fatalf("期望 FormatMismatchError, 实际 %T: %v", err, err)
	}
	t.Logf("结构损坏被正确识别: %v", err)

	// 列索引越界。
	_, err = NewCSR(2, 2, []int{0, 1, 2}, []int{0, 5}, []float64{1, 2})
	var ib *IndexOutOfBoundsError
	if !errors.As(err, &ib) {
		t.Fatalf("期望 IndexOutOfBoundsError, 实际 %T: %v", err, err)
	}
}

func TestResidualAndNorm(t *testing.T) {
	csr, _ := buildDemoCOO(t).ToCSR()
	x := []float64{1, 1, 1}
	ax, _ := MatVec(csr, x)
	rn, err := ResidualNorm(csr, x, ax)
	if err != nil || rn > 1e-14 {
		t.Fatalf("自洽残差应≈0, 得到 %v, err=%v", rn, err)
	}
	if fn := FrobeniusNormSq(csr); math.Abs(fn-(16+1+1+9+4+4+25)) > 1e-12 {
		t.Fatalf("Frobenius 平方错误: %v", fn)
	}
}
