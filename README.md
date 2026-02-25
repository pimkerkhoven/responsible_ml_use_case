# Experiments for `Responsible Machine Learning: Vision, Challenges and Recommendations`

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

This repository contains the experiments for the paper `Responsible Machine Learning: Vision, Challenges and Recommendations` (to be published).
The experiments concern a hypothetical scenario about loan application approval by a bank.
Specifically, an ML system is developed to predict whether or not an individual’s income is above or below 50000 dollars.
This system can then be used as part of the decision-making process for approving loans. 
This system is developed with two different development processes, namely, with the traditional development process focusing on the performance of a system and with a development process incorporating the methods and recommendations proposed in the paper, focusing on additional requirements such as fairness, privacy, and explainability.
<!-- Further details can be found in the paper (Section 8) -->

The systems are developed using the [ACSIncome dataset for California](https://github.com/socialfoundations/folktables), which is derived from US Census data.
Data from 2014 and 2015 is used for training, validating, and testing the system. 
Afterwards, data for 2016 is used in experiments for monitoring a deployed version of the system.
Furthermore, two additional datasets are used.
One that [maps the occupation codes](https://usa.ipums.org/usa/volii/occtooccsoc18.shtml) used by the US Census Bureau to occupation codes used by the US Bureau of Labor Statistics.
The other provides [average annual salaries and average hourly salaries for California](https://www.bls.gov/oes/tables.htm) for each occupation used by the US Bureau of Labor Statistics.
These additional datasets make it possible to augment the dataset with annual and hourly average salaries based on an individual’s occupation.

## Running the experiments

1. Clone the repository
```
git clone https://github.com/pimkerkhoven/ContextualFairness.git
```
2. (Optionally) create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
3. (Optionally) export `PYTHONPATH` (might be necessary to run scripts correctly)
```
export PYTHONPATH="."
```
4. Install requirements
```
pip install -r requirements.txt
```
5. Train models

Three varieties of models are trained.

For the traditional development process, first, some basic models are trained using:

```
python experiments/initial_models.py
```

Based on the initial models, some more optimized models are trained that make use of the additional data sets and perform some hyperparameter optimization:

```
python experiments/improved_models.py
```


For the responsible development process, models are trained using a multi-criteria optimization process:
```
python experiments/responsible_models.py
```


6. Analyze results

To analyze the results, the trained models can be analyzed with MLFlow, which can be started with:
```
mlflow ui
```

In the MLFlow UI, all trained models show up under their respective experiments (i.e., initial, improved, and responsible models). 
Using the UI, for each experiment, a model can be selected and published with one of the following names:
- Initial models
    - initial_logistic_regression
    - initial_decision_tree",
    - initial_naive_bayes
- Improved models
    - improved_logistic_regression
    - improved_decision_tree
    - improved_naive_bayes
- Responsible models
    - responsible_model


Next, these models can be tested by running:

```
python experiments/test_models.py
```

For comparing the `responsible_model` with the best model from the initial & improved models (i.e., the best traditional model), the name of this traditional model should be hard-coded in `experiments/test_models.py` (Line 33).
Currently, `improved_logistic_regression` is hard-coded, as this model was the best traditional model in the paper's experiments.

Note that this requires running `experiments/test_models.py` multiple times. First, to determine the best traditional model and then to compare this model with the `responsible_model`.

<!-- ## Cite paper

```bibtex
@inproceedings{kerkhoven2026vision,
  title={Responsible Machine Learning: Vision, Challenges and Recommendations},
  author={},
  year={}
}

``` -->