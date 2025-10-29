from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.prepare_data.nodes import (
    download_data,
    create_occ_to_sal,
    make_data_splits,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_occ_to_sal,
                inputs=["occ_to_soc", "soc_to_sal", "params:state"],
                outputs="occ_to_sal",
                name="create_occ_to_sal",
            ),
            Node(
                func=make_data_splits,
                inputs=["2014.data", "2015.data"],
                outputs=["train_data", "validation_data", "test_data"],
                name="make_data_splits",
            ),
        ]
    )


base_download_pipeline = Pipeline(
    [
        Node(
            func=download_data,
            inputs=["params:year", "params:state"],
            outputs="data",
            name="download_data",
        ),
        # Node(
        #     func=augment_with_salary_data,
        #     inputs=["data", "occ_to_sal", "params:augment_features"],
        #     outputs="data_with_salary",
        #     name="augment_data_with_salary",
        # ),
    ]
)


def create_2014_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [base_download_pipeline],
        namespace="2014",
        # inputs={"occ_to_sal"},
        parameters={"params:state"},
    )


def create_2015_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [base_download_pipeline],
        namespace="2015",
        # inputs={"occ_to_sal"},
        parameters={"params:state"},
    )


def create_2016_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [base_download_pipeline],
        namespace="2016",
        # inputs={"occ_to_sal"},
        parameters={"params:state"},
    )
