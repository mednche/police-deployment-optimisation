# Determining optimal police patrol deployments: a simulation-based optimisation approach combining Agent-Based Modelling and Genetic Algorithms.


## Project Description

This repository provides complementary code and data for the work undertaken during my PhD.

### Motivations: What problem does it solve? What your model does,

The model was build in a generic manner so as to be applied to any police force. It is made of two components:
- an Agent-Based Model that simulates the dispatching of police patrols throughout a given shift
- a Genetic Algorithm that searches for the optimal number and spatial positioning of police patrols.

- Why you used the technologies you used,

- Some of the challenges you faced and features you hope to implement in the future.

## Structure

The `src` folder contains the model codebase, split into an `ABM` folder -- containing the code for running the ABM on its own -- and a `GA` folder with the code to run the ABM-based (single and multi-objective) optimisation.


The folder `dpd_case_study` contains all the files pertaining to the case study on Detroit Police Department (DPD), Michigan. It contains a `data` folder with the datasets used in the case study, as well as a series of folders containing the code and results for various analyses conducted on the model:
- a sensitivity analysis on the ABM
- a validation of the ABM
- experiments using the ABM on DPD
- experiments using the single and multi-objective GAs on DPD




## Installation

What are the steps required to install your project? Provide a step-by-step description of how to get the development environment running.

Key dependencies include:
Provided yml file for versions that were used.

To run the ABM on its own, 


## Usage

Provide instructions and examples for use. Include screenshots as needed.

To add a screenshot, create an `assets/images` folder in your repository and upload your screenshot to it. Then, using the relative filepath, add it to your README using the following syntax:

    ```md
    ![alt text](assets/images/screenshot.png)
    ```

Show GIF of the ABM running
Expected time of running the GA on multipecessing 

## Licence

MIT

