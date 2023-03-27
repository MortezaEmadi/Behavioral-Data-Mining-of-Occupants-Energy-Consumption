#Identification, Recommendation, & Ranking of Occupants' Behavioral Energy Consumption Patterns

## Project Description:

This project aims to identify building occupants' behavioral patterns
and leverage their energy-saving potential through innovative behavioral
recommendation and ranking systems. A research paper based on the
project is currently under preparation.

The project combines one Multi-Criteria Decision-Making (MCDM) method
and four Machine Learning techniques to develop an intelligent
residential energy-saving solution. A data processing pipeline has been
built to manage the 120GB dataset and select approximately 40GB for the
desired usage.

This repository includes implementations of the following methods/
techniques:

-   HDF5 Library for data storage and organization (Big Data tool)

-   Mutual Information Theory for Feature Selection

-   K-Means & Mean Shift for Two-Level Clustering

-   FP-growth in [Apache Spark™](https://spark.apache.org/) (Big Data tool) for Association Rule
    Mining 
-   Multilayer Perceptron (MLP) as Neural Networks

-   K-Fold Cross Validation & Grid Search for Model Evaluation and
    Parameter Tuning of Neural Networks, respectively

-   Entropy-based TOPSIS for Multi-Criteria Ranking

## Usage Instructions

Dataset Link from Edinburgh Univerity:

<https://datashare.ed.ac.uk/handle/10283/3647>

To execute the modules, you need Python 3.8 and the required libraries.
For running \`spark_fpgrowth.py\`, the Spark engine and Open Java
Development Kit must be installed, or you can use non-commercial free
access to them on the web (e.g., Google Colab).

First, place the dataset folders in the local drive without renaming
them. Below, I list the modules (those with a main function) that should
be executed in order to make use of all the interconnected modules. In
total, 15 modules support this project, utilizing OOP and Modular design
principles in most of them.

###Steps to run:

**1. Run \`feature_selection.py\`**; this prioritizes the non-behavioral
    features of the homes in regard to a reference feature

**2. Run \`clustering.py\`;** to cluster the homes in 2 sequential level
based on their non-behavioral features

**3. Run \`hdfstore.py\`** to store all the ‘needed’ CSVs of dataset in
HDF5 format. Be sure to set the “dataset_path” and “local_temp_path” via
Env. Vars. or in the \`data_store.py\` module.

**4\. Run \`data_preprocess.py\`:** to preprocess recorded data of OEM &
Z-Wave sensor networks separately. (From this point onwards, each step
develops the model for one particular home at a time, with the home's ID
specified in this module.)

**5. Run \`detect_on_off.py\`;** to utilize quantitative ARM, this
module will call many other modules to detect the on/off operational status of appliances in OEM and Z-Wave sensor
networks with innovative technique. Additionally, a complex approach is employed for detecting the
on/off status of **Radiators** and calculating their energy consumption
based on thermodynamic principles (some in step 6).

**6. Run \`spark_fpgrowth.py\`;** this prepares operational data of all
appliances in all timestamps where at least one of the appliances'
on/off status changes. Then, with PySpark installed, it applies the
FP-Growth.

**7. Run \` energy_rule_select.py\`** for selecting the valid rules,
and finding and calculating of anti-rule events’ energies.

**8. Run \`mlp_gridsearch.py\`** which models MLP neural networks for
all the valid rules of each home. Please refer to the \`main\` function
to see how you can specify a particular rule for plotting it’s results
in this module.

**9. Run \`summary.py\`** which calculates the required parameters which
will be use in ranking procedure of entropy-based TOPSIS like, Per
Capita Gas/Electricity Usage of homes.

**10.** Please note that entropy-based TOPSIS was implemented on the
results of this datamining project in Excel oftware. However, it is not
included in this repository to ensure the integrity of the project is
preserved until the publication of its paper. If you need further
detail, please feel free to [email](mailto:info@emadi.me) me.

## License

This project is licensed under the "All rights reserved" license. No
permission is granted to use, distribute, or modify any code or
materials in this repository until further notice from the author. For
more information, please see the [LICENSE](./LICENSE.md) file.

##Reference:

This code is authored by Morteza Emadi. However, some data wrangling components are based on the open-source code provided by Cillian Brewitt, a member of the IDEAL dataset preparation team, with his permission. Cillian's appreciated repository can be found here:

-   <https://github.com/cbrewitt/nilm_fcn>(written by Cillian Brewitt)

## Results Overview:

Please refer to [the result folder in the repository](.result/result_slides.pdf) or my website
<https://emadi.me/#a_ex> for further details of the results and the
research structure.