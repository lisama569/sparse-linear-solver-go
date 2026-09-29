// 示例：用不同存储格式描述同一个稀疏线性方程组，
// 在统一入口下选择 CG / BiCGSTAB 与预处理器求解，并打印可解释的结论。
package main

import (
	"fmt"

	"github.com/lisama569/sparse-linear-solver-go/solver"
	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

func main() {
	// A 为 4 阶 SPD 三对角矩阵（对角 4，次对角 -1）。
	n := 4
	row, col, val := []int{}, []int{}, []float64{}
	for i := 0; i < n; i++ {
		row, col, val = append(row, i), append(col, i), append(val, 4)
		if i+1 < n {
			row = append(row, i, i+1)
			col = append(col, i+1, i)
			val = append(val, -1, -1)
		}
	}

	// 1) 以 COO 构造，再分别转成 CSR / CSC —— 三种格式进入同一个 Solve。
	coo, err := sparse.NewCOO(n, n, row, col, val)
	must(err)
	csr, err := coo.ToCSR()
	must(err)
	csc, err := csr.ToCSC()
	must(err)

	b := []float64{1, 2, 3, 4}
	formats := map[string]sparse.Matrix{"COO": coo, "CSR": csr, "CSC": csc}

	for name, A := range formats {
		res, err := solver.Solve(A, b, solver.Options{
			Method:            solver.MethodCG,
			Preconditioner:    solver.PrecondJacobi,
			Tol:               1e-10,
			AbsTol:            1e-12,
			Criterion:         solver.BothResidual,
			MaxIterations:     100,
			ConditionEstimate: true,
		})
		must(err)
		fmt.Printf("[%s] %s\n        解 x = [", name, res.Summary())
		for i, x := range res.X {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.6f", x)
		}
		fmt.Printf("]  条件数上界≈%.1f\n", res.ConditionEst)
	}

	// 2) 奇异系统：整行为零，结论必须是确定的“奇异”，而不是一串看似正常的数。
	sing, err := sparse.NewCOO(2, 2,
		[]int{0, 0}, []int{0, 1}, []float64{1, 1})
	must(err)
	res, err := solver.Solve(sing, []float64{1, 1}, solver.Options{Method: solver.MethodBiCGSTAB})
	must(err)
	fmt.Printf("\n奇异系统: 状态=%s 原因=%s\n", res.Status, res.FailureReason)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
