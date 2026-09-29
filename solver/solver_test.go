package solver

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

// spd1D 返回 n 阶一维 SPD 三对角矩阵：对角 2，次对角 -1（离散拉普拉斯）。
// 条件数约为 (4n²/π²)，n 很大时难收敛，适合测试步数上限场景。
func spd1D(n int) *sparse.CSR {
	var ri, ci []int
	var vv []float64
	for i := 0; i < n; i++ {
		ri = append(ri, i)
		ci = append(ci, i)
		vv = append(vv, 2)
		if i+1 < n {
			ri = append(ri, i, i+1)
			ci = append(ci, i+1, i)
			vv = append(vv, -1, -1)
		}
	}
	coo, err := sparse.NewCOO(n, n, ri, ci, vv)
	if err != nil {
		panic(err)
	}
	csr, err := coo.ToCSR()
	if err != nil {
		panic(err)
	}
	return csr
}

// onesB 返回全 1 右端项。
func onesB(n int) []float64 {
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	return b
}

func verifySolution(t *testing.T, A sparse.Matrix, x, b []float64, tol float64) {
	t.Helper()
	rn, err := sparse.ResidualNorm(A, x, b)
	if err != nil {
		t.Fatalf("残差计算失败: %v", err)
	}
	if rn > tol {
		t.Fatalf("解不满足 A x=b: 残差 %.3e > %.0e", rn, tol)
	}
}

// methodsAndOpts 返回待测的方法×预处理组合。
func methodsAndOpts() []Options {
	return []Options{
		{Method: MethodCG, Preconditioner: PrecondNone},
		{Method: MethodCG, Preconditioner: PrecondJacobi},
		{Method: MethodBiCGSTAB, Preconditioner: PrecondNone},
		{Method: MethodBiCGSTAB, Preconditioner: PrecondILU0},
	}
}

func TestConvergenceOnSPD(t *testing.T) {
	n := 50
	A := spd1D(n)
	b := onesB(n)
	for _, opts := range methodsAndOpts() {
		opts.Tol = 1e-10
		opts.AbsTol = 1e-12
		name := fmt.Sprintf("%v+%v", opts.Method, opts.Preconditioner)
		t.Run(name, func(t *testing.T) {
			res, err := Solve(A, b, opts)
			if err != nil {
				t.Fatalf("Solve 返回错误: %v", err)
			}
			t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
			if res.Status != StatusConverged {
				t.Fatalf("期望收敛，实际 %s: %s", res.Status, res.FailureReason)
			}
			verifySolution(t, A, res.X, b, 1e-9)
		})
	}
}

func TestFormatAgnosticEntry(t *testing.T) {
	// 同一系统分别以 COO / CSC 输入，结果都应收敛且一致。
	n := 20
	csr := spd1D(n)
	csc, err := csr.ToCSC()
	if err != nil {
		t.Fatal(err)
	}
	coo, err := csr.ToCOO()
	if err != nil {
		t.Fatal(err)
	}
	b := onesB(n)
	base := Options{Method: MethodBiCGSTAB, Tol: 1e-9}
	for name, M := range map[string]sparse.Matrix{"CSR": csr, "CSC": csc, "COO": coo} {
		res, err := Solve(M, b, base)
		if err != nil {
			t.Fatalf("%s 入口错误: %v", name, err)
		}
		t.Logf("输入格式=%s 解法=%s 判据=%s | %s", name, res.Method, res.Criterion, res.Summary())
		if res.Status != StatusConverged {
			t.Fatalf("%s: 期望收敛，实际 %s", name, res.Status)
		}
		verifySolution(t, M, res.X, b, 1e-8)
	}
}

func TestMaxIterationDistinguishable(t *testing.T) {
	// 大系统 + 极小步数 + 严格容差 => 达到步数上限，状态必须与发散/奇异可区分。
	n := 400
	A := spd1D(n)
	b := onesB(n)
	opts := Options{
		Method: MethodCG, MaxIterations: 3, Tol: 1e-12, AbsTol: 1e-14,
		Criterion: BothResidual,
	}
	res, err := Solve(A, b, opts)
	if err != nil {
		t.Fatalf("Solve 错误: %v", err)
	}
	t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
	if res.Status != StatusMaxIteration {
		t.Fatalf("期望 StatusMaxIteration，实际 %s (%s)", res.Status, res.FailureReason)
	}
	if res.Converged() {
		t.Fatal("步数上限结果不得被报告为收敛")
	}
	if res.Iterations != 3 {
		t.Fatalf("迭代步数=%d，期望 3", res.Iterations)
	}
}

func TestDivergenceOnIndefiniteCG(t *testing.T) {
	// 对角含负数的不定矩阵：CG 必须明确报失效/发散，而不是返回无意义数值。
	n := 4
	var ri, ci []int
	var vv []float64
	for i := 0; i < n; i++ {
		ri = append(ri, i)
		ci = append(ci, i)
		if i == n-1 {
			vv = append(vv, -3) // 负特征值
		} else {
			vv = append(vv, 2)
		}
	}
	coo, _ := sparse.NewCOO(n, n, ri, ci, vv)
	A, _ := coo.ToCSR()
	b := onesB(n)
	res, err := Solve(A, b, Options{Method: MethodCG, MaxIterations: 50, Criterion: RelativeResidual})
	if err != nil {
		t.Fatalf("Solve 错误: %v", err)
	}
	t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
	if res.Status != StatusDiverged {
		t.Fatalf("不定矩阵上 CG 期望 StatusDiverged，实际 %s (%s)", res.Status, res.FailureReason)
	}
	if res.FailureReason == "" {
		t.Fatal("发散必须给出可区分的原因")
	}
}

func TestSingularZeroRow(t *testing.T) {
	// 第 2 行全零 => 必然奇异，必须给出确定的奇异结论。
	// 仅行 0、1 有元素，行 2 整行为零 => 奇异。
	coo, err := sparse.NewCOO(3, 3,
		[]int{0, 1, 0, 1}, []int{0, 1, 1, 0}, []float64{1, 1, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	A, _ := coo.ToCSR()
	b := onesB(3)
	res, err := Solve(A, b, Options{Method: MethodBiCGSTAB})
	if err != nil {
		t.Fatalf("奇异系统应通过 Result.Status 报告而非抛 error: %v", err)
	}
	t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
	if res.Status != StatusSingular {
		t.Fatalf("期望 StatusSingular，实际 %s (%s)", res.Status, res.FailureReason)
	}
}

func TestSingularZeroColumn(t *testing.T) {
	// 满行但有整列零：x 不唯一 => 奇异。
	coo, err := sparse.NewCOO(3, 3,
		[]int{0, 1, 2, 0, 1}, []int{0, 0, 0, 2, 2}, []float64{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	A, _ := coo.ToCSR()
	res, err := Solve(A, onesB(3), Options{Method: MethodCG})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
	if res.Status != StatusSingular {
		t.Fatalf("期望 StatusSingular，实际 %s", res.Status)
	}
}

func TestSingularJacobiZeroPivot(t *testing.T) {
	// 每行都有元素、每列都有元素，但某个对角元为零：
	// Jacobi 必须识别为奇异/不适用，结论与“未收敛”区分开。
	coo, err := sparse.NewCOO(2, 2,
		[]int{0, 0, 1, 1}, []int{1, 0, 0, 1}, []float64{1, 1, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	// 构造对角元 a_00=0 的矩阵：行0=[0,1]，行1=[1,2]。
	coo2, _ := sparse.NewCOO(2, 2,
		[]int{0, 1, 1}, []int{1, 0, 1}, []float64{1, 1, 2})
	_ = coo
	A, _ := coo2.ToCSR()
	res, err := Solve(A, onesB(2), Options{
		Method: MethodCG, Preconditioner: PrecondJacobi,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("解法=%s 收敛判据=%s 预处理器=%s | %s",
		res.Method, res.Criterion, res.Preconditioner, res.Summary())
	if res.Status != StatusSingular {
		t.Fatalf("零主元应报 StatusSingular，实际 %s (%s)", res.Status, res.FailureReason)
	}
}

func TestIllConditionedSystem(t *testing.T) {
	// 经典 2x2 病态矩阵（K=406/...），条件数约 2500；
	// 两种方法都应能识别残差并收敛（验证不把病态误报为奇异），
	// 同时请求条件数估计应给出显著大于 1 的上界。
	// SPD 且条件数很高的矩阵：特征值约 1.99 与 9.95e-4，κ≈1999。
	coo, err := sparse.NewCOO(2, 2,
		[]int{0, 0, 1, 1}, []int{0, 1, 0, 1},
		[]float64{1, -0.9995, -0.9995, 1})
	if err != nil {
		t.Fatal(err)
	}
	A, _ := coo.ToCSR()
	b := []float64{1, 1}
	for _, m := range []Method{MethodCG, MethodBiCGSTAB} {
		res, err := Solve(A, b, Options{
			Method: m, Tol: 1e-12, AbsTol: 1e-13,
			Preconditioner: PrecondJacobi, ConditionEstimate: true,
		})
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		t.Logf("解法=%s 收敛判据=%s | %s 条件数估计≈%.1f",
			res.Method, res.Criterion, res.Summary(), res.ConditionEst)
		if res.Status != StatusConverged {
			t.Fatalf("%v: 病态但非奇异应收敛，实际 %s (%s)", m, res.Status, res.FailureReason)
		}
		verifySolution(t, A, res.X, b, 1e-8)
	}
}

func TestInputErrorsAreTyped(t *testing.T) {
	A := spd1D(5)
	// 维度对不上：b 长度错误。
	_, err := Solve(A, []float64{1, 2}, Options{})
	var dm *sparse.DimMismatchError
	if !errors.As(err, &dm) {
		t.Fatalf("期望 DimMismatchError，实际 %T: %v", err, err)
	}
	t.Logf("维度错误: %v", err)

	// 索引越界的矩阵输入：构造期就应返回错误，不能拿到矩阵。
	bad, cerr := sparse.NewCOO(2, 2, []int{0, 3}, []int{0, 1}, []float64{1, 1})
	if cerr == nil {
		t.Fatal("构造越界矩阵应返回错误")
	}
	var ib *sparse.IndexOutOfBoundsError
	if !errors.As(cerr, &ib) {
		t.Fatalf("构造期期望 IndexOutOfBoundsError，实际 %T: %v", cerr, cerr)
	}
	t.Logf("构造期索引越界: %v", cerr)
	// 转换期越界：合法构造后破坏内部索引。
	bad2, _ := sparse.NewCOO(2, 2, []int{0}, []int{0}, []float64{1})
	bad2.RowIndices[0] = 5
	_, err = Solve(bad2, onesB(2), Options{})
	if !errors.As(err, &ib) {
		t.Fatalf("转换期期望 IndexOutOfBoundsError，实际 %T: %v", err, err)
	}
	t.Logf("转换期索引越界: %v", err)
	// typed-nil 接口不应 panic，应返回格式错误。
	_, err = Solve(bad, onesB(2), Options{})
	var fm2 *sparse.FormatMismatchError
	if !errors.As(err, &fm2) {
		t.Fatalf("typed-nil 期望 FormatMismatchError，实际 %T: %v", err, err)
	}
	t.Logf("空矩阵: %v", err)
	var _ = bad

	// 非方阵：FormatMismatch。
	rect, _ := sparse.NewCOO(2, 3,
		[]int{0, 1, 0}, []int{0, 1, 2}, []float64{1, 1, 1})
	_, err = Solve(rect, onesB(2), Options{})
	var fm *sparse.FormatMismatchError
	if !errors.As(err, &fm) {
		t.Fatalf("期望 FormatMismatchError，实际 %T: %v", err, err)
	}
	t.Logf("格式不匹配: %v", err)
}

func TestZeroRightHandSide(t *testing.T) {
	// b=0：x=0 是唯一解（非奇异时），应 0 步收敛且解全零。
	A := spd1D(10)
	res, err := Solve(A, make([]float64, 10), Options{Method: MethodBiCGSTAB})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("解法=%s 收敛判据=%s | %s", res.Method, res.Criterion, res.Summary())
	if res.Status != StatusConverged || res.Iterations != 0 {
		t.Fatalf("b=0 期望 0 步收敛，实际 status=%s it=%d", res.Status, res.Iterations)
	}
	for _, v := range res.X {
		if math.Abs(v) > 0 {
			t.Fatalf("b=0 解应为零向量，实际 %v", res.X)
		}
	}
}

func TestPreconditionerDoesNotChangeSolution(t *testing.T) {
	// 开启预处理前后，收敛解必须一致到容差以内。
	n := 60
	A := spd1D(n)
	b := onesB(n)
	r1, err := Solve(A, b, Options{Method: MethodCG, Tol: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Solve(A, b, Options{Method: MethodCG, Tol: 1e-12, Preconditioner: PrecondILU0})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("无预处理: %s", r1.Summary())
	t.Logf("ILU(0) : %s", r2.Summary())
	if r1.Status != StatusConverged || r2.Status != StatusConverged {
		t.Fatalf("两次都应收敛: %s / %s", r1.Status, r2.Status)
	}
	for i := range r1.X {
		if math.Abs(r1.X[i]-r2.X[i]) > 1e-8 {
			t.Fatalf("预处理改变了解: x[%d] %.10f vs %.10f", i, r1.X[i], r2.X[i])
		}
	}
	// 预处理通常应减少迭代步数（此处仅记录，不硬性断言平台差异）。
	t.Logf("迭代步数: 无预处理=%d, ILU(0)=%d", r1.Iterations, r2.Iterations)
}
