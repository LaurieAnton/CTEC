### CTEC Python Module Overview
This module is an open-source probabalistic implementation of the Couple Traffic, Energy and Traffic (CTEC) model first introduced by Mladen Čičić et al., in 2022, and expanded on in 2023 to include multiple vehicle classes. This module was first used to assess the demand on Fast Charging Service Stations strategically placed along a highway route between the cities of Reykjavik and Akureyri in Iceland and is amenable for use in other settings. 

The files included are as follows:

- The CTEC.py file contains the main engine and needs to be imported when used.
- The postprocessing.py file contains helper functions used in the making of the main tutorial and publication (currently In Review).
- The rvk_auk.ipynb file contains a tutorial on how one can set up a scenario and model each day in a year, seperately.
- The sandbox.ipynb file contains additional test code used to validate convservation of vehicles and energy.
- Lastly, sample data is given to show how it must be formatted if provided via csv.

The user can also import a dataframe of the correct structure as well.
