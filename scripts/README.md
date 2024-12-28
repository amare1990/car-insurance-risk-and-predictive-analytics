# Car Insurance Risk Assessment and Predictive Analysis

> Car Insurance Risk Assessment and Predictive Analysis is a data science project designed to statistically analyze and extract insights from the data from AlphaCare Insurance Solutions (ACIS), a car insurance company in South Africa. This data science project attempts to conduct Exploratory Data Discovery and Analysis (EDA). It finds key insights from the data using statistical and EDA analysis and from the histogram, bar charts, scatter plots, correlation matrix and box plots. It is implemented using Python programming language and various tools including jupyter notebook.

## Built With

- Major languages used: Python3
- Libraries: numpy, pandas, seaborn, matplotlib.pyplot, scikit-learn
- Tools and Technlogies used: jupyter notebook, Git, GitHub, Gitflow, VS code editor.

## Demonstration and Website

[Deployment link](Soon!)

## Getting Started

You can clone my project and use it freely and then contribute to this project.

- Get the local copy, by running `git clone https://github.com/amare1990/car-insurance-risk-and-predictive-analytics.git` command in the directory of your local machine.
- Go to the repo main directory, run `cd car-insurance-risk-and-predictive-analytics` command
- Create python environment by running `python3 -m venv venm-name`
- Activate it by running:
- `source venv-name/bin/activate` on linux os command prompt if you use linux os
- `myenv\Scripts\activate` on windows os command prompt if you use windows os.

- After that you have to install all the necessary Python libraries and tools by running `pip install -r requirements.txt`
- To run this project, run `jupyter notebook` command from the main directory of the repo

### Prerequisites

- You have to install Python (version 3.8.10 minimum), pip, git, vscode.

### Dataset

 - `MachineLearningRating_cleaned.txt` text file is used as a dataset. The text dataset is separated by '`|`'.
 - Run `df = pandas.read_csv("/path to your file/MachineLearningRating_cleaned.txt", delimiter="|")`

### Project Requirements
- Git, GitHub setup, dev setup including GitHub action,  Adding `pylint' in the GitHub workflows
- Statistical and EDA analysis on the data, ploting
- Data Version Control setup and commits

#### GitHub Action
- Go to the main directory of this repo, create paths, `.github/workflows`. And then add `pylint` linters
- Make it to check when Pull request is created
- Run `pylint scripts/script_name.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/script_name.py` to automatically fix some linters errors

### Statistical and EDA
In this portion of the task the following analysis has been conducted.

- Data Summary:
    Calculate descriptive statistics for numerical features like TotalPremium and TotalClaim.
    Verify proper formatting of data types for columns (e.g., categorical, dates).

- Data Quality Check:
    Identify and address missing values.

- Univariate Analysis:
    Plot histograms for numerical variables and bar charts for categorical ones.

- Bivariate/Multivariate Analysis:
    Explore relationships between TotalPremium, TotalClaims, and ZipCode using scatter plots and correlation matrices.

- Data Comparison:
    Analyze trends in insurance cover, premium, and auto make across different regions.

-Outlier Detection:
    Use box plots to identify outliers in numerical data.

- Visualization:
    Create 3 visually appealing plots to highlight key insights from the analysis.

#### Data Version Control
- The main purpose of this task is to allow learners exercise in applying version control on data
- Run `dvc init` command to initialize dvc
- Create directory where data will be stored, applied modificationa, and then tracked
- Setup the local remote storage by running `dvc remote add -d localstorage dvc directory` command
- Run `dvc add "your path/dvc directory/"` command for the dvc to track it
- Create dvc pipeline execution scripts for perprocessing
- Run `dvc repro` in the main directory of the repo to automatically run all the pipeline stages

> #### You can gain more insights by running the jupter notebook and view plots.


### More information
- You can refer to [this link](Coming Soon!) to gain more insights about the presentation of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/car-insurance-risk-and-predictive-analytics/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project
- Thanks to work on the Microverse's README.md template

## 📝 License

This project is [MIT](./LICENSE) licensed.
