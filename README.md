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
- a tuning experiment for the RSS parameter of the single-objective GA
- results of the single and multi-objective GAs applied to DPD




## Installation

What are the steps required to install your project? To get the development environment running.

Key dependencies include:
- `osmnx` for downlading the road network of a police force and calculating fastest routes
- `networkX`for manipulating the road network graph generated by osmnx
- `geopandas` for manipulating the spatial dataframes generated by osmnx
- `imageio` for producing a GIF of the ABM
- multiprocessing to be used on multiple cores.
Provided yml file for versions that were used.

To run the ABM, it is necessary to aquire the following files and place them in the `data` folder: 

1. `G.pickle`: Download the road network using osmnx and save the graph as `G.pickle`. May need to provide a simple jupyter notebook for that.
2. `patrol_beats.shp`: Acquire the patrol beat shapefile and save as `patrol_beats.shp`.
3. Optional: for producing a GIF, the code requires a `precincts.shp` and a `stations.csv`, although these are optional.


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

