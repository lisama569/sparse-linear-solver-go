package solver

import (
	"errors"

	"github.com/lisama569/sparse-linear-solver-go/sparse"
)

func asDim(err error, target **sparse.DimMismatchError) bool {
	return errors.As(err, target)
}
