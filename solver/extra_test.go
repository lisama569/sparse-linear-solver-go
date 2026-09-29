package solver

import (
	"fmt"
	"testing"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// nonsymTriDiag 返回 n 阶非对称但可逆的三对角矩阵：
// 对角 4，下次对角 -1，上次对角 -2。
func nonsymTriDiag(n int) *sparse.CSR {
	var ri, ci []int
	var vv []float64
	for i := 0; i < n; i++ {
		ri, ci, vv = append(ri, i), append(ci, i), append(vv, 4.0)
		if i+1 < n {
			ri = append(ri, i, i+1)
			ci = append(ci, i+1, i)
			vv = append(vv, -2.0, -1.0)
		}
	}
	coo, err := sparse.NewCOO(n, n, ri, ci, vv)
	if err != nil {
		panic(err)
	}
	a, err := coo.ToCSR()
	if err != nil {
		panic(err)
	}
	return a
}

func TestBiCGSTABOnNonsymmetric(t *testing.T) {
	n := 100
	A := nonsymTriDiag(n)
	b := onesB(n)
	for _, pc := range []PrecondType{PrecondNone, PrecondILU0} {
		res, err := Solve(A, b, Options{
			Method: MethodBiCGSTAB, Preconditioner: pc,
			Tol: 1e-10, AbsTol: 1e-12, MaxIterations: 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("解法=%s 预处理器=%s 收敛判据=%s | %s",
			res.Method, res.Preconditioner, res.Criterion, res.Summary())
		if res.Status != StatusConverged {
			t.Fatalf("非对称系统应收敛，实际 %s: %s", res.Status, res.FailureReason)
		}
		verifySolution(t, A, res.X, b, 1e-8)
	}
}

func TestCriterionVariants(t *testing.T) {
	// 三种判据都能给出明确结论；绝对判据只看 ‖r‖₂。
	n := 30
	A := spd1D(n)
	b := onesB(n)
	cases := []struct {
		name      string
		criterion ResidualCriterion
	}{
		{"相对残差", RelativeResidual},
		{"绝对残差", AbsoluteResidual},
		{"双判据", BothResidual},
	}
	for _, c := range cases {
		res, err := Solve(A, b, Options{
			Method: MethodCG, Criterion: c.criterion,
			Tol: 1e-6, AbsTol: 1e-8,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("判据=%s(%s) 解法=%s | %s", c.name, res.Criterion, res.Method, res.Summary())
		if res.Status != StatusConverged {
			t.Fatalf("%s 下期望收敛，实际 %s", c.name, res.Status)
		}
	}
}

func TestInitialGuessHonored(t *testing.T) {
	// 给定接近真解的初值应显著减少步数；错误长度的初值必须报维度错误。
	n := 80
	A := spd1D(n)
	b := onesB(n)

	fromZero, err := Solve(A, b, Options{Method: MethodCG, Tol: 1e-10})
	if err != nil {
		t.Fatal(err)
	}
	good := append([]float64(nil), fromZero.X...)
	for i := range good {
		good[i] *= 1.001 // 真解附近的小扰动
	}
	near, err := Solve(A, b, Options{Method: MethodCG, Tol: 1e-10, InitialGuess: good})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("零初值: %s", fromZero.Summary())
	t.Logf("近解初值: %s", near.Summary())
	if near.Status != StatusConverged || near.Iterations > fromZero.Iterations {
		t.Fatalf("近解初值应不增加迭代步数: %d vs %d", near.Iterations, fromZero.Iterations)
	}

	_, err = Solve(A, b, Options{Method: MethodCG, InitialGuess: []float64{1, 2}})
	var dm *sparse.DimMismatchError
	if !asDim(err, &dm) {
		t.Fatalf("错误长度初值应报 DimMismatchError，实际 %v", err)
	}
	t.Logf("初值维度错误: %v", err)
}

func TestStatusStringsAreStable(t *testing.T) {
	// 锁定四种终态与方法/判据的可读名称，保证输出可复现、可解释。
	if got := fmt.Sprintf("%v|%v|%v|%v",
		StatusConverged, StatusMaxIteration, StatusDiverged, StatusSingular); got != "已收敛|达到步数上限仍未收敛|数值发散或解算失效|奇异或明显病态，解不存在或不唯一" {
		t.Fatalf("终态名称不稳定: %s", got)
	}
}
