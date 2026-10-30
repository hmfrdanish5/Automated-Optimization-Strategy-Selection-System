"""Run the Review-1 audit, search-efficiency table, and validation outputs."""

from evaluation.reproducibility import (
    model_artifact_report,
    print_review1_banner,
    run_unit_test_counts,
    write_reproducibility_report,
)
from evaluation.validation import write_validation


if __name__ == "__main__":
    write_validation()
    print(write_reproducibility_report())
    print(model_artifact_report())
    counts = run_unit_test_counts()
    print_review1_banner(test_passed=counts["passed"], test_failed=counts["failed"])
