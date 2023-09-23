"""An ambiguous mnist data set"""

import csv

import datasets
import numpy as np
from datasets.tasks import ImageClassification

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2207.10495,
  doi = {10.48550/ARXIV.2207.10495},
  url = {https://arxiv.org/abs/2207.10495},
  author = {Weiss, Michael and Gómez, André García and Tonella, Paolo},
  title = {A Forgotten Danger in DNN Supervision Testing: Generating and Detecting True Ambiguity},
  publisher = {arXiv},
  year = {2022}
}
"""

_DESCRIPTION = """\
The images were created such that they have an unclear ground truth, 
i.e., such that they are similar to multiple - but not all - of the datasets classes.
Robust and uncertainty-aware models should be able to detect and flag these ambiguous images.
As such, the dataset should be merged / mixed with the original dataset and we
provide such 'mixed' splits for convenience. Please refer to the dataset card for details.
"""

_HOMEPAGE = "https://github.com/testingautomated-usi/ambiguous-datasets"
_LICENSE = "https://raw.githubusercontent.com/testingautomated-usi/ambiguous-datasets/main/LICENSE"

_VERSION = "0.1.0"
_URL = f"https://github.com/testingautomated-usi/ambiguous-datasets/releases/download/v{_VERSION}/"

_URLS = {
    "train": "mnist-test.csv",
    "test": "mnist-test.csv",
}

_NAMES = [
    "T - shirt / top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class MnistAmbiguous(datasets.GeneratorBasedBuilder):
    """An ambiguous mnist data set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="mninst_ambiguous",
            version=datasets.Version(_VERSION),
            description=_DESCRIPTION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                    "text_label": datasets.Value("string"),
                    "p_label": datasets.Sequence(datasets.Value("float32"), length=10),
                    "is_ambiguous": datasets.Value("bool"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {key: _URL + fname for key, fname in _URLS.items()}
        downloaded_files = dl_manager.download(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name="train_mixed",
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train_mixed",
                },
            ),
            datasets.SplitGenerator(
                name="test_mixed",
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test_mixed",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw form."""

        def _gen_amb_images():
            with open(filepath) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for i, row in enumerate(spamreader):
                    if i == 0:
                        continue

                    det_label = int(row[7])
                    class_1, class_2 = int(row[3]), int(row[4])
                    p_1, p_2 = float(row[5]), float(row[6])
                    text_label = f"p({_NAMES[class_1]})={p_1:.2f}, p({_NAMES[class_2]})={p_2:.2f}"

                    p_label = [0.0] * 10
                    p_label[class_1] = p_1
                    p_label[class_2] = p_2

                    image = np.array(row[9:], dtype=np.uint8).reshape(28, 28)

                    yield i, {"image": image, "label": det_label,
                              "text_label": text_label, "p_label": p_label, "is_ambiguous": True}

        if split == "test" or split == "train":
            yield from _gen_amb_images()

        elif split == "test_mixed" or split == "train_mixed":

            nominal_samples = []
            nom_split = "test" if split == "test_mixed" else "train"
            nominal_dataset = datasets.load_dataset("mnist", split=nom_split)
            for x in nominal_dataset:
                nominal_samples.append({
                    "image": np.array(x["image"]),
                    "label": x["label"],
                    "text_label": f"p({_NAMES[x['label']]})=1",
                    "p_label": [1.0 if i == x["label"] else 0.0 for i in range(10)],
                    "is_ambiguous": False
                })

            ambiguous_samples = list([x for i, x in _gen_amb_images()])
            all_samples = nominal_samples + ambiguous_samples
            np.random.RandomState(42).shuffle(all_samples)

            for i, x in enumerate(all_samples):
                yield i, x


if __name__ == '__main__':
    dataset = MnistAmbiguous()
    dataset.download_and_prepare()
    ds = dataset.as_dataset()
    print(ds)
