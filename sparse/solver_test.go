package sparse

import (
	"math"
	"testing"
)

// 严格对角占优的 SPD 三对角矩阵（COO 构造）。
func spdTridiag(t *testing.T, n int, diag, off float64) *COO {
	t.Helper()
	var es []Entry
	for i := 0; i < n; i++ {
		es = append(es, Entry{i, i, diag})
		if i+1 < n {
			es = append(es, Entry{i, i + 1, off}, Entry{i + 1, i, off})
		}
	}
	m, err := NewCOO(n, n, es)
	if err != nil {
		t.Fatalf("NewCOO: %v", err)
	}
	return m
}

func ones(n int) []float64 {
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	return b
}

// logResult 按要求在单测日志中打印解法与收敛判据。
func logResult(t *testing.T, name string, cfg Config, r Result) {
	t.Helper()
	precond := "none"
	if cfg.Preconditioner != nil {
		precond = cfg.Preconditioner.Name()
	}
	t.Logf("case=%s method=%s tol=%.1e maxIter=%d precond=%s -> status=%s iters=%d relRes=%.3e reason=%q",
		name, name, cfg.Tolerance, cfg.MaxIterations, precond, r.Status, r.Iterations, r.Residual, r.Reason)
}

// 所有解法在三种存储格式下都应收敛到同一真解。
func TestSolversConvergeAcrossFormats(t *testing.T) {
	n := 50
	coo := spdTridiag(t, n, 4, 1)
	formats := map[string]Matrix{"COO": coo, "CSR": coo.ToCSR(), "CSC": coo.ToCSC()}
	methods := []Method{Jacobi{}, GaussSeidel{}, CG{}}
	cfg := Config{Tolerance: 1e-9, MaxIterations: 5000}

	for fname, a := range formats {
		// 构造真解 x* = (1,2,...,n)，b = A x*
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = float64(i + 1)
		}
		b, err := a.MatVec(xs)
		if err != nil {
			t.Fatalf("MatVec: %v", err)
		}
		for _, m := range methods {
			r := Solve(a, b, m, cfg)
			logResult(t, m.Name()+"@"+fname, cfg, r)
			if r.Status != Converged {
				t.Fatalf("%s on %s: status=%s reason=%q", m.Name(), fname, r.Status, r.Reason)
			}
			for i := range xs {
				if math.Abs(r.X[i]-xs[i]) > 1e-5 {
					t.Fatalf("%s on %s: x[%d]=%v, want %v", m.Name(), fname, i, r.X[i], xs[i])
				}
			}
		}
	}
}

// 预处理应减少迭代步数，且不改变解。
func TestPreconditionerSpeedsUpCG(t *testing.T) {
	n := 200
	// 对角元变化剧烈的 SPD 矩阵，Jacobi 预处理才有加速空间
	var es []Entry
	for i := 0; i < n; i++ {
		es = append(es, Entry{i, i, 2.0 + float64(i%13)})
		if i+1 < n {
			es = append(es, Entry{i, i + 1, 0.5}, Entry{i + 1, i, 0.5})
		}
	}
	a, err := NewCOO(n, n, es)
	if err != nil {
		t.Fatal(err)
	}
	b := ones(n)
	cfgPlain := Config{Tolerance: 1e-10, MaxIterations: 10000}
	cfgPre := Config{Tolerance: 1e-10, MaxIterations: 10000, Preconditioner: &JacobiPreconditioner{}}
	plain := Solve(a, b, CG{}, cfgPlain)
	pre := Solve(a, b, CG{}, cfgPre)
	logResult(t, "cg-plain", cfgPlain, plain)
	logResult(t, "cg-jacobi-precond", cfgPre, pre)
	if plain.Status != Converged || pre.Status != Converged {
		t.Fatalf("plain=%s pre=%s", plain.Status, pre.Status)
	}
	if pre.Iterations > plain.Iterations {
		t.Errorf("preconditioned CG took %d iters > plain %d iters", pre.Iterations, plain.Iterations)
	}
	for i := range b {
		if math.Abs(plain.X[i]-pre.X[i]) > 1e-6 {
			t.Fatalf("preconditioner changed the solution at %d: %v vs %v", i, plain.X[i], pre.X[i])
		}
	}
}

// 难以收敛：严格限制步数，必须报告 MaxIterationsReached 且与 Failed 可区分。
func TestHardToConvergeReportsMaxIterations(t *testing.T) {
	n := 500
	a := spdTridiag(t, n, 2.0, 1.0) // 条件数 O(n^2)，收敛慢
	b := ones(n)
	cfg := Config{Tolerance: 1e-12, MaxIterations: 5}
	r := Solve(a, b, CG{}, cfg)
	logResult(t, "cg-hard", cfg, r)
	if r.Status != MaxIterationsReached {
		t.Fatalf("status=%s, want max-iterations-reached", r.Status)
	}
	if r.Iterations != cfg.MaxIterations {
		t.Errorf("iterations=%d, want %d", r.Iterations, cfg.MaxIterations)
	}
	if r.Reason == "" {
		t.Error("Reason should explain non-convergence")
	}
}

// 奇异矩阵：零行使 Jacobi/Gauss-Seidel 报零主元失效。
func TestSingularMatrixZeroPivot(t *testing.T) {
	m, err := NewCOO(3, 3, []Entry{{0, 0, 2}, {1, 1, 2}}) // 第 3 行全零
	if err != nil {
		t.Fatal(err)
	}
	b := []float64{1, 1, 1}
	for _, method := range []Method{Jacobi{}, GaussSeidel{}} {
		r := Solve(m, b, method, Config{})
		logResult(t, method.Name()+"-singular", Config{}, r)
		if r.Status != Failed {
			t.Fatalf("%s: status=%s, want failed", method.Name(), r.Status)
		}
		if r.Reason == "" {
			t.Errorf("%s: missing failure reason", method.Name())
		}
	}
}

// 病态矩阵：Hilbert 型，CG 在宽松步数内无法达到极严判据，状态必须明确。
func TestIllConditionedMatrix(t *testing.T) {
	n := 12
	var es []Entry
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			es = append(es, Entry{i, j, 1.0 / float64(i+j+1)})
		}
	}
	h, err := NewCOO(n, n, es)
	if err != nil {
		t.Fatal(err)
	}
	b := ones(n)
	cfg := Config{Tolerance: 1e-15, MaxIterations: 100}
	r := Solve(h, b, CG{}, cfg)
	logResult(t, "cg-hilbert", cfg, r)
	// 病态 + 极严判据：要么数值失效，要么步数耗尽，绝不允许谎报收敛。
	if r.Status == Converged && r.Residual > cfg.Tolerance {
		t.Fatalf("reported converged but residual %.3e > tol %.3e", r.Residual, cfg.Tolerance)
	}
	if r.Status != Converged && r.Reason == "" {
		t.Error("non-converged result must carry a reason")
	}
}

// 非对称矩阵用 CG：残差发散，必须以 Failed 收场而非返回无意义数值。
func TestCGRejectsNonSymmetric(t *testing.T) {
	m, _ := NewCOO(2, 2, []Entry{{0, 0, 1}, {0, 1, 2}, {1, 0, 0}, {1, 1, 1}})
	r := Solve(m, []float64{1, 1}, CG{}, Config{Tolerance: 1e-12, MaxIterations: 100000})
	logResult(t, "cg-nonsym", Config{Tolerance: 1e-12, MaxIterations: 100000}, r)
	if r.Status != Failed {
		t.Fatalf("status=%s, want failed for non-SPD input", r.Status)
	}
}

// 对称不定矩阵（特征值含负数）触发 CG 的 p^T A p <= 0 崩溃，必须 Failed。
func TestCGBreakdownOnIndefinite(t *testing.T) {
	m, _ := NewCOO(2, 2, []Entry{{0, 0, 1}, {0, 1, 2}, {1, 0, 2}, {1, 1, 1}})
	r := Solve(m, []float64{1, 0}, CG{}, Config{Tolerance: 1e-12, MaxIterations: 100})
	logResult(t, "cg-indefinite", Config{Tolerance: 1e-12, MaxIterations: 100}, r)
	if r.Status != Failed {
		t.Fatalf("status=%s, want failed for indefinite matrix", r.Status)
	}
}

// 维度不匹配与非法右端必须以 Failed 给出可区分原因。
func TestInvalidInputsFailWithReason(t *testing.T) {
	a := spdTridiag(t, 3, 4, 1)
	cases := []struct {
		name string
		b    []float64
	}{
		{"dim-mismatch", []float64{1, 2}},
		{"nan-rhs", []float64{1, math.NaN(), 1}},
		{"inf-rhs", []float64{1, math.Inf(1), 1}},
	}
	for _, tc := range cases {
		r := Solve(a, tc.b, CG{}, Config{})
		logResult(t, "cg-"+tc.name, Config{}, r)
		if r.Status != Failed || r.Reason == "" {
			t.Errorf("%s: status=%s reason=%q, want failed with reason", tc.name, r.Status, r.Reason)
		}
	}
}

// 预处理器遇到零对角（奇异）必须让求解以 Failed 收场。
func TestPreconditionerSetupFailurePropagates(t *testing.T) {
	m, _ := NewCOO(2, 2, []Entry{{0, 1, 1}, {1, 0, 1}}) // 对角全零
	r := Solve(m, []float64{1, 1}, CG{}, Config{Preconditioner: &JacobiPreconditioner{}})
	logResult(t, "cg-precond-singular", Config{}, r)
	if r.Status != Failed || r.Reason == "" {
		t.Fatalf("status=%s reason=%q, want failed with reason", r.Status, r.Reason)
	}
}
