from matplotlib import pyplot as plt

from src.rml_vision_usecase.pipelines.train_model.make_radar_plot import (
    create_radar_plot,
)

data = [
    (1, 1, 1, 1, 1),
    (0.5, 3, 4, 5, 6),
    (0.5, 3, 4, 5, 6),
]


for i, (precision, accuracy, privacy, fairness, explainability) in enumerate(data):
    radar_plot = create_radar_plot(
        data=[precision, accuracy, privacy, fairness, explainability],
        labels=["Precision", "Accuracy", "Privacy", "Fairness", "Explainability"],
        color_index=i,
    )

    plt.show()
