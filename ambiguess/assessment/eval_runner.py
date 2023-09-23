"""The quantitative evaluation of AmbiGuess."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import uncertainty_wizard as uwiz
from uncertainty_wizard import ProblemType

from assessment import eval_model


def create_empty_result_table():
    """Create an empty result table."""
    training_sets = ['mixed-ambiguous', 'clean']
    case_study = ['mnist', 'fmnist']
    test_set = ['nominal', 'ambiguous']
    table_index = pd.MultiIndex.from_product([training_sets, case_study, test_set],
                                             names=['Training Set', 'Case S.', 'Test Set'])
    df = pd.DataFrame(index=table_index, columns=['Top-1 Acc', 'Top-2 Acc', 'Top-Pair Acc', 'Entropy'])
    return df


def _format_cell(x):
    if x is None or x == 'nan' or np.isnan(x):
        return "n.a."
    return "{:.2f}".format(x)


def write_full_csv_results(result_table: pd.DataFrame, folder: str):
    """Write the full results to a csv file."""
    result_table.to_csv(folder + "/complete_results.csv")
    print("Stored results as csv in: " + folder + "/ambiguity_complete_results.csv")


def _table_block_for_ds(result_table: pd.DataFrame, dataset: str):
    cn = result_table.loc[("clean", dataset, 'nominal')]
    ca = result_table.loc[("clean", dataset, 'ambiguous')]
    mn = result_table.loc[("mixed-ambiguous", dataset, 'nominal')]
    ma = result_table.loc[("mixed-ambiguous", dataset, 'ambiguous')]

    if dataset == "mukhoti":
        assert ca['Top-Pair Acc'] == "n.a." and ma['Top-Pair Acc'] == "n.a.", "Mukhoti does not have Top-Pair Acc"
        ca['Top-Pair Acc'] = "not calculable"
        ma['Top-Pair Acc'] = "not calculable"

    block = rf"""
\multirow{{2}}{{*}}{{mixed-ambiguous}}   & ambiguous  & {ma['Top-1 Acc']} & {ma['Top-2 Acc']} & {ma['Top-Pair Acc']} & {ma['Entropy']}\\
                                         & nominal    & {mn['Top-1 Acc']} & {mn['Top-2 Acc']} & n.a. & {mn['Entropy']}\\
\multirow{{2}}{{*}}{{clean}}             & ambiguous  & {ca['Top-1 Acc']} & {ca['Top-2 Acc']} & {ca['Top-Pair Acc']} & {ca['Entropy']}\\
                                         & nominal    & {cn['Top-1 Acc']} & {cn['Top-2 Acc']} & n.a. & {cn['Entropy']}\\
    """
    return block


_large_table_header = r"""
\begin{table}
\centering
\begin{tabular}{@{}llcccc@{}}
\textbf{Training Set}            & \textbf{Test Set} & \textbf{Top-1 Acc} & \textbf{Top-2 Acc} & \textbf{Top-Pair Acc} & \textbf{Entropy} \\\midrule
"""


def _large_table_end(model: str):
    model_str = f"({model})" if model else ""
    label_suffix = f"_{model.replace(' ', '').replace('.', '').lower()}" if model else ""
    return rf"""
\bottomrule
\end{{tabular}}
\caption{{Evaluation of Ambiguity {model_str}}}
\label{{tab:ambiguity_res{label_suffix}}}
\end{{table}}
"""


def _large_table_subtitle(val: str):
    return rf"\multicolumn{{6}}{{c}}{{\textit{{{val}}}}}      \vspace{{3px}} \\"


def print_full_results_latex(result_table: pd.DataFrame, folder: str, run):
    """Prints the full results, including baseline and clean ds, as latex table"""
    tex_table = result_table.copy()
    tex_table = tex_table.applymap(_format_cell)

    if run is not None:
        model = eval_model.ambi_eval_architecture(run).name()
    else:
        model = None

    tex = "\n".join([
        _large_table_header,
        _large_table_subtitle("Fashion MNIST"),
        _table_block_for_ds(tex_table, "fmnist"),
        r"\midrule",
        _large_table_subtitle("MNIST"),
        _table_block_for_ds(tex_table, "mnist"),
        r"\midrule",
        _large_table_subtitle(r"Baseline for mnist: AmbiguousMNIST by Mukhoti et. al.~\cite{Mukhoti2021}"),
        _table_block_for_ds(tex_table, "mukhoti"),
        _large_table_end(model)
    ])

    file_suffix = f"_{model.replace(' ', '').replace('.','').lower()}" if model else ""
    with open(folder + f"/full_ambiguity_tex_table{file_suffix}.tex", "w") as f:
        f.write(tex)


def print_results_as_latex_table(result_table: pd.DataFrame, folder: str):
    """Print a subset of the results as a latex table."""
    # Note: The original table remains unaltered (otherwise, pass inplace=True)
    tex_table = result_table.drop('clean', level=0)
    # Uncomment this line if you want to see the mukhoti evaluation in the latex table
    tex_table = tex_table.drop('mukhoti', level=1)

    # tex_table.index.droplevel(0)
    tex_table = tex_table.droplevel(0)

    # To_latex has formatters options, but its not working as expected for floats
    tex_table = tex_table.applymap(_format_cell)

    tex = tex_table.to_latex(
        column_format='llcccc',
        # caption = "Accuracies using nominal or ambiguous test data.",
        label="tab:ambiguity_res",
    )
    tex = tex.replace("\end{table}",
                      """
          \caption{Prediction Performance using nominal or ambiguous test data.
          \end{table}""")

    # Multiline Headers, to reduce table width
    tex = tex.replace("Top-1 Acc", r"\begin{tabular}[x]{@{}c@{}}Top-1\\Acc\end{tabular} ")
    tex = tex.replace("Top-2 Acc", r"\begin{tabular}[x]{@{}c@{}}Top-2\\Acc\end{tabular} ")
    tex = tex.replace("Top-Pair Acc", r"\begin{tabular}[x]{@{}c@{}}Top-Pair\\Acc\end{tabular} ")

    # print("=========================================================")
    # print("Accuracy table (nominal or ambiguous test data). ")
    # print("=========================================================")
    #
    # print(tex)
    #
    # print("=========================================================")
    # print("======= Tex table was also stored on file system ========")
    # print("=========================================================")

    with open(folder + "/ambiguity_tex_table.tex", "w") as f:
        f.write(tex)


def evaluate(model: uwiz.models.StochasticSequential,
             dataset: str,
             clean: bool,
             nominal_test_data: Tuple[np.ndarray, np.ndarray],
             ambiguous_test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
             result_table: pd.DataFrame):
    """Evaluate the AmbiGuess outputs on a provided model."""
    clean_str = 'clean' if clean else 'mixed-ambiguous'
    res_nom = _evaluate_atomic(model=model,
                               x=nominal_test_data[0],
                               y_single_pred=nominal_test_data[1],
                               prob_labels=None)
    result_table.at[(clean_str, dataset, 'nominal'),
    ('Top-1 Acc', 'Top-2 Acc', 'Top-Pair Acc', 'Entropy')] = res_nom

    res_amb = _evaluate_atomic(model=model,
                               x=ambiguous_test_data[0],
                               y_single_pred=ambiguous_test_data[1],
                               prob_labels=ambiguous_test_data[2])
    result_table.at[(clean_str, dataset, 'ambiguous'),
    ('Top-1 Acc', 'Top-2 Acc', 'Top-Pair Acc', 'Entropy')] = res_amb


def _top2accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    """Accuracy where the (single) correct class has to be amongst the top-2 prediction"""
    assert y_pred[0].shape == y_true.shape
    assert y_pred[1].shape == y_true.shape
    first_correct = y_pred[0] == y_true
    second_correct = y_pred[1] == y_true
    return (np.count_nonzero(first_correct) + np.count_nonzero(second_correct)) / y_true.shape[0]


def _pair_accuracy(top_2_res, prob_labels: np.ndarray):
    """Accuracy where the top-2 predictions equal the two ambiguous classes

    This metric is applicable to ambiguous data, where the ambiguity is between exactly
    two classes.
    """
    pair_correct_count = 0
    for s in range(top_2_res[0].shape[0]):
        if prob_labels[s][top_2_res[0][s]] > 0 and prob_labels[s][top_2_res[1][s]] > 0:
            pair_correct_count += 1
        assert np.sum(prob_labels[s] > 0) == 2
    return pair_correct_count / prob_labels.shape[0]


def _evaluate_atomic(model: uwiz.models.StochasticSequential,
                     x: np.ndarray,
                     y_single_pred: np.ndarray,
                     prob_labels: Optional[np.ndarray]):
    x = x.astype("float32") / 255
    with warnings.catch_warnings(record=True) as w:
        quantifiers = ["softmax_entropy", TopTwoPredictions()]
        entropy_res, top2_res = model.predict_quantified(x, quantifier=quantifiers)
        # We expect the following warning. If any other warning is raised, the assertion will fail.
        assert "You are predicting both confidences and uncertainties." in str(w[-1].message)

    top_1_acc = _accuracy(entropy_res[0], y_single_pred)
    top_2_acc = _top2accuracy(top2_res[0], y_single_pred)
    avg_entropy = np.mean(entropy_res[1])
    if prob_labels is None or np.any(np.sum(prob_labels > 0, axis=1) == 1):
        # Contains non-ambiguous data, so we can not compute the pair accuracy (case for Mukhoti)
        prob_labels = None
    pair_accuracy = _pair_accuracy(top2_res[0], prob_labels) if prob_labels is not None else None
    return (top_1_acc, top_2_acc, pair_accuracy, avg_entropy)


def _accuracy(prediction, det_y_true):
    num_samples = prediction.shape[0]
    return np.count_nonzero((prediction == det_y_true)) / num_samples


class TopTwoPredictions(uwiz.quantifiers.Quantifier):
    """uwiz quantifier to get the 2 prediction with highest probability"""

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["custom::toptwo"]

    # docstr-coverage:inherited
    @classmethod
    def is_confidence(cls) -> bool:
        return True

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        first = np.argmax(nn_outputs, axis=1)
        # first_values = nn_outputs[np.arange(first.shape[0]), first]
        nn_outputs[np.arange(first.shape[0]), first] = -1
        second = np.argmax(nn_outputs, axis=1)
        # second_values = nn_outputs[np.arange(first.shape[0]), second]
        return (first, second), None
