# Titanic: Spark and Machine Learning from Disaster <!-- omit in toc -->

[![license: GPL3][license-img]][license-link]
![latest commit](https://img.shields.io/github/last-commit/Silemo/dic-2023-manfredi-meneghin)

## About <!-- omit in toc -->

Titanic: Spark and Machine Learning from Disaster is a project of **Data Intensive Computing (ID2221)**, course of Data Science held at KTH - Royal Institute of Technology (Period 1, 2023/2024).

**Professor** Amir Hossein Payberah

### The Team <!-- omit in toc -->

* [Giovanni Manfredi](https://github.com/Silemo)
* [Sebastiano Meneghin](https://github.com/SebastianoMeneghin)

## Table of content <!-- omit in toc -->

- [Specifications](#specifications)
- [Documentation](#documentation)
- [Running the project](#running-the-project)
- [Software used](#software-used)
  

## Specifications

The goal of this project is to **process and query data** regarding the [historical Titanic wreckage](https://www.kaggle.com/c/titanic) and implement a predictive analysis to **classify** which of the passenger on board will survive the tragedy, based on various parameters (name, age, gender, social status, etc.). The structure in **Spark** for pre-processing and querying for the model training will be the main output of this project.

The initial project proposal can be found [here][proj-prop-link].

## Documentation

A full and complete report of the classification algorithm can be found [here][final-report-link].

## Running the project

The project was developed on the Databricks cloud data platform. The free license we used is the [Databricks Community Edition](https://docs.databricks.com/en/getting-started/community-edition.html). Once you have signed up and logged in in the platform, the first thing you need to do is entering the *Data Science and Engineering* section, going into the menu *Compute* and create and start your own server cluster.

Once the server cluster is running, you have to go in the menu *Workspace*, create a new folder where to run your project and import in it the notebook *main.scala* present on our repository, through the option *Import > File*. When you have opened it, open the tab *Connect* and load your notebook on the cluster by selecting the cluster server you have created.

Now, into the menu *Data* you have to import the file *dataset.csv* present in our repository, through the option *Create Table $>$ Upload File*. Once you have selected the file *dataset.csv*, you need to press on *Create Table with UI* then select the cluster you have created above. Now, in the *table preview* window, you need to select *first row is header* and *infer schema * and \textbf{to rename the file as *dataset*. Now, pressing on *Create Table* you will have the access to the necessary data to run the notebook.

Now you can return on the notebook *main.scala* and run the all project, by pressing on the *Run All* button.

## Software used

**Databricks Community edition** - main developing environment

**GitKraken** - github

**OneDrive** - file sharing

**Overleaf** - LaTeX editor

**Visual Studio Code** - Markdown editor



<!--Links of the document-->
[license-img]: https://img.shields.io/badge/license-GPL--3.0-blue
[license-link]: https://github.com/Silemo/tiw-2022-manfredi-meneghin/blob/master/LICENSE
[proj-prop-link]: https://github.com/Silemo/dic-2023-manfredi-meneghin/tree/main/deliverables/Task4_ProjectProposal_ManfrediMeneghin.pdf
[final-report-link]: https://github.com/Silemo/dic-2023-manfredi-meneghin/tree/main/deliverables/Task4_Report_ManfrediMeneghin.pdf