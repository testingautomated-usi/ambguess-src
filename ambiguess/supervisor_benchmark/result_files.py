"""Utils to write and receive results from the file system as pandas dataframes."""
import os.path
from typing import Tuple, Dict

import numpy as np
import pandas
import pandas as pd

from supervisor_benchmark import model_architectures

CASE_STUDIES = ['mnist', 'fmnist']
TEST_SET_TYPES = ['ambiguous', 'adversarial', 'corrupted', 'invalid']


# docstr-coverage:excused `private`
def __create_new_result_file():
    table_index = pd.MultiIndex.from_product([CASE_STUDIES, TEST_SET_TYPES])
    df = pd.DataFrame(columns=table_index)
    df.index.name = 'supervisor'
    return df


# docstr-coverage:excused `private`
def __get_results_dataframe(metric: str,
                            run_id: int,
                            artifacts_folder: str) -> pd.DataFrame:
    file_path = os.path.join(__run_folder(artifacts_folder, run_id), f"results_{metric}.pickle")
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return __create_new_result_file()


# docstr-coverage:excused `private`
def __persist_datasframe(df: pd.DataFrame,
                         metric: str,
                         run_id: int,
                         artifacts_folder: str) -> None:
    folder = __run_folder(artifacts_folder, run_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_pickle(os.path.join(folder, f"results_{metric}.pickle"))


# docstr-coverage:excused `private`
def __run_folder(artifacts_folder, run_id):
    return os.path.join(artifacts_folder, "supervisor_benchmark", "runs", f"{run_id}", )


def register_results(dataset_and_split: Tuple[str, str],
                     supervisor: str,
                     metric: str,
                     value: float,
                     run_id: int,
                     artifacts_folder: str) -> None:
    """Adds a new result to the per-run dataframe (persisted in artifacts folder)"""
    df = __get_results_dataframe(metric, run_id, artifacts_folder)
    df.at[supervisor, dataset_and_split] = value
    __persist_datasframe(df, metric, run_id, artifacts_folder)


def plot(artifacts_folder: str,
         run_id: int) -> None:
    """Creates nice CSV and TEX tables"""
    df = __get_results_dataframe('auc_roc', run_id, artifacts_folder)
    df.to_csv(os.path.join(__run_folder(artifacts_folder, run_id), "auc_roc.csv"))

    # latex_str = _to_latex(df)
    # with open(os.path.join(__run_folder(artifacts_folder, run_id), "auc_roc.tex"), 'w') as f:
    #     f.write(latex_str)


def _del_if_exists(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def _format_approach_means(x):
    if x is None or x == 'nan' or np.isnan(x):
        return "n.a."
    return "{:.2f}".format(x).lstrip('0')


def _format_appraoch_stds(x):
    if x is None or x == 'nan' or np.isnan(x):
        return ""
    return r"<std-start>" + "{:.2f}".format(x).lstrip('0') + "<std-end>"


def plot_aggregate(artifacts_folder: str):
    """Aggregates all runs and creates nice CSV and TEX tables"""

    # Collect all dataframes
    all_dfs: Dict[int, pandas.DataFrame] = dict()
    for root, dirs, files in os.walk(f"{artifacts_folder}/supervisor_benchmark/runs"):
        for file in files:
            if file == "results_auc_roc.pickle":
                run = int(root.split("/")[-1])
                all_dfs[run] = pd.read_pickle(os.path.join(root, file))

    concat_df = pd.concat(all_dfs.values())
    aggregated_df = concat_df.groupby(level=0)
    # Means .CSV and .TEX
    means = aggregated_df.mean()
    means.to_csv(os.path.join(artifacts_folder, "supervisor_benchmark", "mean_auc_roc.csv"))
    with open(os.path.join(artifacts_folder, "supervisor_benchmark", "mean_auc_roc.tex"), 'w') as f:
        f.write(_to_latex(means,
                          caption="Supervisors performance at discriminating nominal from high-uncertainty inputs "
                                  "(AUC-ROC), averaged over all architectures.",
                          label="tab:mean_auc_roc"))

    # Per-Architecture Analysis, including statistical stuff
    by_architecture = dict()
    for run, df in all_dfs.items():
        architecture = type(model_architectures.architecture_choice(run)).__name__
        if architecture not in by_architecture:
            by_architecture[architecture] = []
        by_architecture[architecture].append(df)

    all_means = []
    all_stds = []
    for architecture, dfs in by_architecture.items():
        # Create folder if not exists
        arch_folder = os.path.join(artifacts_folder, "supervisor_benchmark", "per-architectures", architecture)
        if not os.path.exists(arch_folder):
            os.makedirs(arch_folder)

        # Make sure there are 5 runs per architecture (hardcoded value in stability string),
        #   warn if not
        if len(dfs) != 5:
            print(f"Pay attention: {architecture} has {len(dfs)} runs, but 5 are expected.")
            print("Make sure you have 5 runs per architecture to reproduce paper results.")

        architecture_df = pd.concat(dfs).groupby(level=0)
        # Per-Architecture Mean .CSV
        means = aggregated_df.mean()
        means.to_csv(os.path.join(arch_folder, "mean_auc_roc.csv"))
        all_means.append(means)
        # Per-Architecture Std .CSV
        stds = architecture_df.std()
        stds.to_csv(os.path.join(arch_folder, "std_auc_roc.csv"))
        all_stds.append(stds)

        # Combined Table (Entries are mean +- std)
        _save_per_architecture_results_tex(architecture, means, stds, artifacts_folder)

    # Stability String .TEX
    with open(os.path.join(artifacts_folder, "supervisor_benchmark", "stability_statement.tex"), 'w') as f:
        f.write(_stability_str(all_means, all_stds))


def _save_per_architecture_results_tex(architecture: str, means: pandas.DataFrame, stds: pandas.DataFrame,
                                       artifacts_folder: str):
    means_formatted = means.applymap(_format_approach_means)
    stds_formatted = stds.applymap(_format_appraoch_stds)
    approach_table = "" + means_formatted + " +- " + stds_formatted

    tex_str = _to_latex(approach_table,
                        caption=f"Supervisor's performance at discriminating nominal from high-uncertainty inputs "
                                f"(AUC-ROC), for the {architecture} architecture.",
                        label=f"tab:auc_roc_{architecture.lower().replace(' ', '_')}")
    # Fix "n.a." fields
    tex_str = tex_str.replace("n.a. +- ", "n.a.")
    # So save space, make shorter approach names (we have the subheaders to disambiguate between MC-D and Ensemble)
    #   We need the additional space in these tables as we included the standard deviations.
    tex_str = tex_str.replace("Max. Softmax", "Max. SM.")
    tex_str = tex_str.replace("Softmax Entropy", "SM. Ent.")
    tex_str = tex_str.replace("MC-Dropout (VR)", "VR")
    tex_str = tex_str.replace("MC-Dropout (MS)", "MS")
    tex_str = tex_str.replace("MC-Dropout (MI)", "MI")
    tex_str = tex_str.replace("MC-Dropout (PE)", "PE")
    tex_str = tex_str.replace("Deep Ensemble (MS)", "MS")
    tex_str = tex_str.replace("Deep Ensemble (MI)", "MI")
    tex_str = tex_str.replace("Deep Ensemble (PE)", "PE")
    tex_str = tex_str.replace("Autoencoder", "Autoenc.")

    # We still need to make font size a little smaller
    tex_str = tex_str.replace(r"\begin{table*}[t]", r"\begin{table*}[t]\small")

    # Hack to make the "std" text italic
    tex_str = tex_str.replace("<std-start>", r"\newline\textit{")
    tex_str = tex_str.replace("<std-end>", r"}")

    file_name = f"mean_auc_roc_{architecture.lower().replace(' ', '')}.tex"
    with open(os.path.join(artifacts_folder, "supervisor_benchmark", file_name), 'w') as f:
        f.write(tex_str)


def _stability_str(all_means, all_stds):
    means = pd.concat(all_means)
    stds = pd.concat(all_stds)

    stability_str_intro = r"""
    % AUTOMATICALLY GENERATED PARAGRAPH. PLEASE TELL ME IF YOU CHANGE ANYTHING,
    % AND I WILL UPDATE THE SCRIPT.
    
    \paragraph{Stability of Results}
    We found that our results are barely sensitive to random influences due to training: """

    larger_than_005 = np.sum(stds.fillna(0).values > 0.05)
    larger_005_and_small_aucroc = np.logical_and(means.to_numpy() < 0.9, stds.to_numpy() > 0.05).sum()
    stability_str = """
    % 8 * 16 - 6 (adversarial / deep ensembles which are n.a.)
    Out of 488 reported mean AUC-ROCs (4 architectures, 8 test sets, 16 MPs, averaged over 5 runs) 
    most of them showed a negligible standard deviation:
    The average observed standard deviation was {avg_std:.3f}, the highest one was {max_dev:.3f}, 
    only {larger_than_002} were larger than 0.02, 
    only {larger_than_005} were larger than 0.05""".format(
        avg_std=np.mean(stds.values[np.logical_not(np.isnan(stds.values))]),
        max_dev=np.max(stds.fillna(0).values),
        larger_than_002=np.sum(stds.fillna(0).values > 0.02),
        larger_than_005=larger_than_005
    )
    if larger_than_005 == larger_005_and_small_aucroc:
        stability_str += """, all of which corresponded to results with low mean AUC-ROC (\\textless 0.9). 
    The latter differences do not influence the overall observed tendencies.
        """
    else:
        stability_str += "."
    res = (stability_str_intro + stability_str)
    return res


def _to_latex(df: pd.DataFrame,
              caption: str, label: str) -> str:
    try:

        sub_df = df.copy()
        sub_df = sub_df.reindex([
            "Max. Softmax",
            "PCS",
            "Softmax Entropy",
            "DeepGini",
            "MC-Dropout (VR)",
            "MC-Dropout (MS)",
            "MC-Dropout (MI)",
            "MC-Dropout (PE)",
            "Deep Ensemble (MS)",
            "Deep Ensemble (MI)",
            "Deep Ensemble (PE)",
            "Dissector",
            "DSA",
            "LSA",
            "MDSA",
            "Autoencoder",
        ])
    except KeyError as e:
        print(e)
        raise ValueError(f"Cannot generate result table as some results have not yet been computed."
                         f"Run replication package for supervisor 'all'.")

    sub_df = sub_df.fillna("n.a.")

    latex_str = sub_df.to_latex(multicolumn_format="c",
                                column_format="l|YYYY|YYYY",
                                index_names=False,
                                float_format=lambda x: "%.2f" % x)

    table_def_start = r"""
    \begin{table*}[t]
    """

    table_def_end = r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n" + r"\end{table*}"

    latex_str = table_def_start + latex_str + table_def_end

    latex_str.replace("Deep Ensemble", "Ensemble")

    latex_str = latex_str.replace("ambiguous", "amb.") \
        .replace("corrupted", "corr.") \
        .replace("adversarial", "adv.") \
        .replace("invalid", "inv.")

    def subheader(_subheader: str, inclide_first_linebreak: bool = True):
        return "".join([
            r"\vspace{2px}\\" if inclide_first_linebreak else "",
            "\n",
            r"\multicolumn{9}{l}{\small \textit{",
            _subheader,
            r"}}\vspace{2px}\\"
        ])

    latex_str = latex_str.replace("\\midrule\nMax. Softmax",
                                  r"\midrule" + "\n" + subheader("Plain Softmax Supervisors", False)
                                  + r"Max. Softmax")
    latex_str = latex_str.replace("\\\\\nMC-Dropout (VR)",
                                  subheader(
                                      "Monte-Carlo Dropout Supervisors (Softmax-based, except for VR)") + r"MC-Dropout (VR)")
    latex_str = latex_str.replace("\\\\\nDeep Ensemble (MS)",
                                  subheader("Deep Ensemble Supervisors (Softmax-based)") + r"Deep Ensemble (MS)")
    latex_str = latex_str.replace("\\\\\nDissector",
                                  subheader("Other Supervisors") + r"Dissector")

    latex_str = latex_str.replace(r"\begin{tabular}", r"\begin{tabularx}{\linewidth}")
    latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabularx}")

    latex_str = r"""
    \newcolumntype{Y}{>{\centering\arraybackslash}X}
    """ + latex_str

    latex_str = r"""
    % ATTENTION: This is an automatically generated file. Do not edit!

    """ + latex_str

    return latex_str
