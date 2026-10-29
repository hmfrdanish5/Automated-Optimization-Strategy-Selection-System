"""Train the Review-1 models and compare them with the rule baseline.

The input matrix is only FEATURE_COLUMNS. Labels are best_strategy.
Running this module trains once and saves the fitted models. It does not
retrain while an optimizer is being applied.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from dataset.dataset_builder import DATASET_PATH, EXPANDED_DATASET_PATH
from dataset.diversity import diversity_report, feature_signature, format_diversity_report
from dataset.examples import load_sample_programs
from dataset.generator import (
    EXPANDED_GENERATED_COUNT,
    GENERATED_PROGRAM_COUNT,
    GENERATION_SEED,
    generate_diverse_programs,
    generate_programs,
)
from evaluation.classification import (
    CLASS_ORDER,
    SHORT_CLASS_NAMES,
    accuracy,
    confusion,
    macro_f1,
    report_text,
)
from evaluation.label_policy import NO_IMPROVEMENT
from evaluation.metrics import instruction_gain, weighted_cost_gain
from features.extractor import FEATURE_COLUMNS
from optimizers.registry import apply_strategy
from selection.classifiers import (
    StrategyClassifier,
    decision_tree,
    load_model,
    random_forest,
    save_model,
)
from selection.config import (
    DATASET_SEED,
    FOREST_PATH,
    MODEL_RANDOM_STATE,
    REVIEW_SPLIT,
    SPLIT_RANDOM_STATE,
    STRICT_SPLIT,
    TEST_SIZE,
    TREE_PATH,
)
from selection.rule_based import predict_rule
from selection.splitting import LabeledExample, SplitInfo, split_examples
from tac.program import TACProgram


def load_dataset_frame(path=DATASET_PATH) -> pd.DataFrame:
    """Read the CSV. Callers must select FEATURE_COLUMNS before training."""
    frame = pd.read_csv(path)
    missing = [column for column in (*FEATURE_COLUMNS, "best_strategy", "program_id") if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The only legal model input."""
    return frame.loc[:, list(FEATURE_COLUMNS)].copy()


def target_labels(frame: pd.DataFrame) -> pd.Series:
    """The measured strategy label. This is the training target, not a feature."""
    return frame["best_strategy"].astype(str)


def load_labeled_examples(frame: pd.DataFrame | None = None) -> list[LabeledExample]:
    """Pair each CSV row with the TAC program that produced it.

    Programs are rebuilt with the dataset seed. Their ids must match the CSV.
    """
    table = frame if frame is not None else load_dataset_frame()
    programs = [(program_id, program) for program_id, program in load_sample_programs()]
    programs.extend(
        (program_id, program)
        for program_id, program, _category in generate_programs(
            GENERATED_PROGRAM_COUNT,
            GENERATION_SEED,
        )
    )
    if [program_id for program_id, _program in programs] != list(table["program_id"]):
        raise ValueError("dataset rows are not in the same order as the seeded programs")
    examples: list[LabeledExample] = []
    for (_program_id, program), row in zip(programs, table.to_dict(orient="records")):
        features = {column: row[column] for column in FEATURE_COLUMNS}
        examples.append(
            LabeledExample(
                program_id=str(row["program_id"]),
                program=program,
                features=features,
                label=str(row["best_strategy"]),
            )
        )
    return examples


def load_expanded_examples(frame: pd.DataFrame | None = None) -> list[LabeledExample]:
    """Pair the expanded CSV with the programs that produced it."""
    table = frame if frame is not None else load_dataset_frame(EXPANDED_DATASET_PATH)
    programs = [(program_id, program) for program_id, program in load_sample_programs()]
    programs.extend(
        (program_id, program)
        for program_id, program, _category in generate_diverse_programs(
            EXPANDED_GENERATED_COUNT,
            GENERATION_SEED,
        )
    )
    return _examples_from_table(table, programs)


def _examples_from_table(table: pd.DataFrame, programs: list[tuple[str, TACProgram]]) -> list[LabeledExample]:
    if [program_id for program_id, _program in programs] != list(table["program_id"]):
        raise ValueError("dataset rows are not in the same order as the seeded programs")
    examples: list[LabeledExample] = []
    for (_program_id, program), row in zip(programs, table.to_dict(orient="records")):
        features = {column: row[column] for column in FEATURE_COLUMNS}
        examples.append(
            LabeledExample(
                program_id=str(row["program_id"]),
                program=program,
                features=features,
                label=str(row["best_strategy"]),
            )
        )
    return examples


def _frame(examples: list[LabeledExample]) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame([example.features for example in examples], columns=FEATURE_COLUMNS)
    labels = pd.Series([example.label for example in examples], name="best_strategy")
    return features, labels


def measured_gains(program: TACProgram, strategy: str) -> tuple[int, int]:
    """Instruction-count gain and weighted-cost gain of one named strategy.

    `no_improvement` applies no optimizer, so both gains are zero. The two
    numbers stay separate: a strategy can lower operation cost without
    deleting an instruction.
    """
    if strategy == NO_IMPROVEMENT:
        return 0, 0
    optimized = apply_strategy(program.copy(), strategy)
    return (
        instruction_gain(program, optimized),
        weighted_cost_gain(program, optimized),
    )


def train_review_models(
    examples: list[LabeledExample] | None = None,
    mode: str = REVIEW_SPLIT,
    save: bool = True,
) -> dict[str, object]:
    """Fit both models and score the held-out programs.

    Models are written only when `save` is true. The expanded evaluation
    saves the strict split, which is the split used to discuss generalization.
    """
    labeled = examples if examples is not None else load_labeled_examples()
    train, test, split = split_examples(labeled, mode)
    x_train, y_train = _frame(train)
    x_test, y_test = _frame(test)

    tree = decision_tree().fit(x_train, y_train)
    forest = random_forest().fit(x_train, y_train)
    if save:
        save_model(tree, TREE_PATH)
        save_model(forest, FOREST_PATH)

    tree_predictions = tree.predict(x_test)
    forest_predictions = forest.predict(x_test)
    rule_predictions = [predict_rule(example.features) for example in test]
    actual = list(y_test)

    practical = []
    for example, rule_name, tree_name, forest_name in zip(
        test, rule_predictions, tree_predictions, forest_predictions
    ):
        best_instruction, best_weighted = measured_gains(example.program, example.label)
        rule_instruction, rule_weighted = measured_gains(example.program, rule_name)
        tree_instruction, tree_weighted = measured_gains(example.program, tree_name)
        forest_instruction, forest_weighted = measured_gains(example.program, forest_name)
        practical.append(
            {
                "program_id": example.program_id,
                "ground_truth": example.label,
                "actual_best_instruction_gain": best_instruction,
                "actual_best_weighted_cost_gain": best_weighted,
                "rule_selected_strategy": rule_name,
                "rule_instruction_gain": rule_instruction,
                "rule_weighted_cost_gain": rule_weighted,
                "decision_tree_strategy": tree_name,
                "decision_tree_instruction_gain": tree_instruction,
                "decision_tree_weighted_cost_gain": tree_weighted,
                "random_forest_strategy": forest_name,
                "random_forest_instruction_gain": forest_instruction,
                "random_forest_weighted_cost_gain": forest_weighted,
            }
        )

    return {
        "examples": labeled,
        "train": train,
        "test": test,
        "split": split,
        "tree": tree,
        "forest": forest,
        "actual": actual,
        "rule_predictions": rule_predictions,
        "tree_predictions": tree_predictions,
        "forest_predictions": forest_predictions,
        "practical": practical,
        "importances": forest.feature_importances(),
        "saved": save,
    }


def train_expanded_evaluation(
    examples: list[LabeledExample] | None = None,
) -> dict[str, dict[str, object]]:
    """Train on the expanded dataset with both split modes.

    The saved model files are the strict feature-grouped fit.
    """
    labeled = examples if examples is not None else load_expanded_examples()
    return {
        REVIEW_SPLIT: train_review_models(labeled, REVIEW_SPLIT, save=False),
        STRICT_SPLIT: train_review_models(labeled, STRICT_SPLIT, save=True),
    }


def format_expanded_report(results: dict[str, dict[str, object]]) -> str:
    """Review-1 and strict reports, printed separately."""
    review = results[REVIEW_SPLIT]
    strict = results[STRICT_SPLIT]
    rows = [
        {
            "program_id": example.program_id,
            "best_strategy": example.label,
            **example.features,
        }
        for example in review["examples"]
    ]
    diversity = diversity_report(rows)
    parts = [
        "========================================",
        "Expanded dataset diversity",
        "========================================",
        format_diversity_report(diversity),
        "",
        _format_poor_classes(diversity.class_counts, diversity.row_count),
        "",
        format_training_report(review),
        "",
        format_training_report(strict),
    ]
    return "\n".join(parts)


def _format_groups(split: SplitInfo) -> str:
    if split.train_groups is None:
        return "Feature groups: row-level split, groups were not held out."
    return (
        f"Feature groups in train: {split.train_groups}\n"
        f"Feature groups in test: {split.test_groups}\n"
        "Feature-vector overlap between train/test: "
        f"{split.overlap_rows}"
    )


def _format_importances(importances: dict[str, float]) -> str:
    return "\n".join(f"  {name}: {importances[name]:.4f}" for name in FEATURE_COLUMNS)


def _format_poor_classes(class_counts: dict[str, int], total: int) -> str:
    lines = ["Classes with fewer than 5 samples:"]
    poor = [name for name in CLASS_ORDER if class_counts.get(name, 0) < 5]
    if not poor:
        smallest = min(CLASS_ORDER, key=lambda name: class_counts.get(name, 0))
        lines.append(
            f"  none. Smallest class is {smallest} with {class_counts.get(smallest, 0)} of {total}."
        )
    else:
        for name in poor:
            lines.append(f"  {name}: {class_counts.get(name, 0)}")
    return "\n".join(lines)


def format_distribution(labels: list[str]) -> str:
    """Count labels in CLASS_ORDER, including classes with zero rows."""
    counts = Counter(labels)
    return "\n".join(f"  {name}: {counts.get(name, 0)}" for name in CLASS_ORDER)


def _gain_cell(instruction_gain_value: object, weighted_gain_value: object) -> str:
    return f"{int(instruction_gain_value):>3}/{int(weighted_gain_value):<3}".rjust(12)


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_training_report(result: dict[str, object]) -> str:
    """The text printed after a training run."""
    examples = result["examples"]
    train = result["train"]
    test = result["test"]
    split: SplitInfo = result["split"]
    tree: StrategyClassifier = result["tree"]
    forest: StrategyClassifier = result["forest"]
    actual: list[str] = result["actual"]
    sections = [
        "========================================",
        "Dataset",
        "========================================",
        f"Samples: {len(examples)}",
        f"Features: {len(FEATURE_COLUMNS)}",
        f"Dataset seed: {DATASET_SEED}",
        f"Train/test random_state: {SPLIT_RANDOM_STATE}",
        f"Model random_state: {MODEL_RANDOM_STATE}",
        f"Test size: {TEST_SIZE}",
        f"Evaluation mode: {split.mode}",
        "",
        "These results are preliminary because the dataset is synthetic and relatively small.",
        "The strict feature-grouped split prevents identical feature signatures from crossing the train/test boundary.",
        "",
        f"Total samples: {len(examples)}",
        f"Unique feature vectors: {len({feature_signature(example.features) for example in examples})}",
        f"Duplicate feature rows: {len(examples) - len({feature_signature(example.features) for example in examples})}",
        f"Test rows whose feature vector also appears in training: {split.overlap_rows}",
        "",
        "Class distribution:",
        format_distribution([example.label for example in examples]),
        "",
        f"no_improvement support in the full dataset: "
        f"{sum(example.label == NO_IMPROVEMENT for example in examples)}",
        "",
        f"Train samples: {len(train)}",
        "Train class distribution:",
        format_distribution([example.label for example in train]),
        "",
        f"Test samples: {len(test)}",
        "Test class distribution:",
        format_distribution([example.label for example in test]),
        "",
        f"Split: {split.explanation}",
        _format_groups(split),
        "",
        "========================================",
        "Decision Tree",
        "========================================",
        f"Accuracy: {accuracy(actual, result['tree_predictions']):.3f}",
        f"Macro F1: {macro_f1(actual, result['tree_predictions']):.3f}",
        "",
        "Classification report:",
        report_text(actual, result["tree_predictions"]).rstrip(),
        "",
        "Confusion matrix (rows are the measured label, columns are the prediction):",
        _format_matrix(confusion(actual, result["tree_predictions"])),
        "",
        "Tree text:",
        tree.export_tree_text().rstrip(),
        "",
        "========================================",
        "Random Forest",
        "========================================",
        f"Accuracy: {accuracy(actual, result['forest_predictions']):.3f}",
        f"Macro F1: {macro_f1(actual, result['forest_predictions']):.3f}",
        "",
        "Classification report:",
        report_text(actual, result["forest_predictions"]).rstrip(),
        "",
        "Confusion matrix (rows are the measured label, columns are the prediction):",
        _format_matrix(confusion(actual, result["forest_predictions"])),
        "",
        "Feature importance:",
        _format_importances(result["importances"]),
        "",
        "========================================",
        "Rule-Based",
        "========================================",
        f"Accuracy: {accuracy(actual, result['rule_predictions']):.3f}",
        f"Macro F1: {macro_f1(actual, result['rule_predictions']):.3f}",
        "",
        "Classification report:",
        report_text(actual, result["rule_predictions"]).rstrip(),
        "",
        "Confusion matrix (rows are the measured label, columns are the prediction):",
        _format_matrix(confusion(actual, result["rule_predictions"])),
        "",
        _format_example(test[0], forest),
        "",
        _format_comparison(result["practical"]),
        "",
        "========================================",
        "Saved models",
        "========================================",
        _format_saved(result, test),
    ]
    return "\n".join(sections)


def format_prediction_demo(result: dict[str, object], limit: int = 4) -> str:
    """Show features, both predictions, forest probabilities, and measured gains."""
    blocks = ["========================================", "Review-1 prediction demo", "========================================"]
    forest: StrategyClassifier = result["forest"]
    tree: StrategyClassifier = result["tree"]
    for example in _demo_examples(result["test"], result["practical"], limit):
        tree_prediction = tree.predict_one(example.features)
        forest_prediction = forest.predict_one(example.features)
        best_instruction, best_weighted = measured_gains(example.program, example.label)
        tree_instruction, tree_weighted = measured_gains(example.program, tree_prediction.strategy)
        forest_instruction, forest_weighted = measured_gains(example.program, forest_prediction.strategy)
        blocks.extend(
            [
                "",
                f"Program: {example.program_id}",
                example.program.to_tac(),
                "Features:",
                *[f"  {column}: {example.features[column]}" for column in FEATURE_COLUMNS],
                f"Decision Tree: {tree_prediction.strategy} (confidence {tree_prediction.confidence:.2f})",
                f"Random Forest: {forest_prediction.strategy} (confidence {forest_prediction.confidence:.2f})",
                "Random Forest probabilities:",
                *[
                    f"  {name}: {forest_prediction.probabilities.get(name, 0.0):.2f}"
                    for name in CLASS_ORDER
                ],
                f"Ground truth: {example.label}",
                f"Ground-truth instruction gain: {best_instruction}",
                f"Ground-truth weighted-cost gain: {best_weighted}",
                f"Decision Tree instruction gain: {tree_instruction}",
                f"Decision Tree weighted-cost gain: {tree_weighted}",
                f"Random Forest instruction gain: {forest_instruction}",
                f"Random Forest weighted-cost gain: {forest_weighted}",
            ]
        )
    return "\n".join(blocks)


def _format_matrix(matrix) -> str:
    short = [SHORT_CLASS_NAMES[name] for name in CLASS_ORDER]
    header = " ".join(f"{name:>6}" for name in short)
    lines = [
        "Columns and rows: " + ", ".join(f"{short_name}={name}" for name, short_name in zip(CLASS_ORDER, short)),
        f"{'':>6} {header}",
    ]
    for label, row in zip(short, matrix):
        cells = " ".join(f"{int(value):>6}" for value in row)
        lines.append(f"{label:>6} {cells}")
    return "\n".join(lines)


def _format_example(example: LabeledExample, forest: StrategyClassifier) -> str:
    prediction = forest.predict_one(example.features)
    lines = [
        "========================================",
        "Random Forest Example Prediction",
        "========================================",
        f"Program: {example.program_id}",
        f"Predicted: {prediction.strategy}",
        f"Confidence: {prediction.confidence:.2f}",
        "",
        "Confidence is the predicted class's probability. It is not a guarantee",
        "that the optimizer will reduce the program.",
        "",
        "Probabilities:",
    ]
    for name in CLASS_ORDER:
        lines.append(f"  {name}: {prediction.probabilities.get(name, 0.0):.2f}")
    return "\n".join(lines)


def _format_comparison(practical: list[dict[str, int | str]]) -> str:
    def column(name: str) -> list[int]:
        return [int(row[name]) for row in practical]

    lines = [
        "========================================",
        "Rule vs Decision Tree vs Random Forest",
        "========================================",
        "Classification metrics are above. The means below are what the",
        "chosen optimizer actually did on the held-out TAC.",
        "Instruction gain and weighted-cost gain are reported separately.",
        "",
        f"Mean ground-truth instruction gain: {_mean(column('actual_best_instruction_gain')):.3f}",
        f"Mean ground-truth weighted-cost gain: {_mean(column('actual_best_weighted_cost_gain')):.3f}",
        f"Mean rule instruction gain: {_mean(column('rule_instruction_gain')):.3f}",
        f"Mean rule weighted-cost gain: {_mean(column('rule_weighted_cost_gain')):.3f}",
        f"Mean decision tree instruction gain: {_mean(column('decision_tree_instruction_gain')):.3f}",
        f"Mean decision tree weighted-cost gain: {_mean(column('decision_tree_weighted_cost_gain')):.3f}",
        f"Mean random forest instruction gain: {_mean(column('random_forest_instruction_gain')):.3f}",
        f"Mean random forest weighted-cost gain: {_mean(column('random_forest_weighted_cost_gain')):.3f}",
        "",
        "Each held-out program. A gain cell is instruction gain / weighted-cost gain.",
        f"{'program':<18} {'truth':<6} {'rule':<6} {'tree':<6} {'forest':<6} "
        f"{'best gain':>12} {'rule gain':>12} {'tree gain':>12} {'forest gain':>12}",
    ]
    for row in practical:
        lines.append(
            f"{row['program_id']:<18} "
            f"{SHORT_CLASS_NAMES[str(row['ground_truth'])]:<6} "
            f"{SHORT_CLASS_NAMES[str(row['rule_selected_strategy'])]:<6} "
            f"{SHORT_CLASS_NAMES[str(row['decision_tree_strategy'])]:<6} "
            f"{SHORT_CLASS_NAMES[str(row['random_forest_strategy'])]:<6} "
            f"{_gain_cell(row['actual_best_instruction_gain'], row['actual_best_weighted_cost_gain'])} "
            f"{_gain_cell(row['rule_instruction_gain'], row['rule_weighted_cost_gain'])} "
            f"{_gain_cell(row['decision_tree_instruction_gain'], row['decision_tree_weighted_cost_gain'])} "
            f"{_gain_cell(row['random_forest_instruction_gain'], row['random_forest_weighted_cost_gain'])}"
        )
    return "\n".join(lines)


def _demo_examples(
    test: list[LabeledExample],
    practical: list[dict[str, object]],
    limit: int,
) -> list[LabeledExample]:
    """Prefer programs whose selectors disagree, then cover different labels."""
    by_id = {example.program_id: example for example in test}
    chosen: list[LabeledExample] = []
    seen: set[str] = set()
    for row in practical:
        names = {
            row["ground_truth"],
            row["rule_selected_strategy"],
            row["decision_tree_strategy"],
            row["random_forest_strategy"],
        }
        if len(names) > 1 and row["program_id"] not in seen:
            chosen.append(by_id[str(row["program_id"])])
            seen.add(str(row["program_id"]))
    for example in test:
        if example.label not in {item.label for item in chosen} and example.program_id not in seen:
            chosen.append(example)
            seen.add(example.program_id)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _format_saved(result: dict[str, object], test: list[LabeledExample]) -> str:
    if not result["saved"]:
        return (
            "This split was not written to disk.\n"
            "The files below are the strict feature-grouped models when train.py has been run."
        )
    return (
        f"Decision tree: {TREE_PATH}\n"
        f"Random forest: {FOREST_PATH}\n"
        "\n"
        "Reloaded models are checked against the test predictions below.\n"
        + _format_reload_check(test, result["tree_predictions"], result["forest_predictions"])
    )


def _format_reload_check(
    test: list[LabeledExample],
    tree_predictions: list[str],
    forest_predictions: list[str],
) -> str:
    reloaded_tree = load_model(TREE_PATH)
    reloaded_forest = load_model(FOREST_PATH)
    features, _labels = _frame(test)
    tree_matches = reloaded_tree.predict(features) == tree_predictions
    forest_matches = reloaded_forest.predict(features) == forest_predictions
    return (
        f"Reloaded decision tree matches: {tree_matches}\n"
        f"Reloaded random forest matches: {forest_matches}"
    )
