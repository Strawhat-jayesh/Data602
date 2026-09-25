# DATA 602 Project (title TBD)
 
## Team
 
Jessie Gordon, Apoorva Manthena, Nick McPhaull, Huayin Shao, Jayesh Sharma
 
## What this is
 
Short-horizon forecasting of dissolved oxygen in the Chesapeake Bay using 15-minute
readings from fixed water quality sensors.
 
Two questions we are working toward:
 
1. How far ahead can dissolved oxygen be predicted before the model stops beating a
   naive persistence baseline?
2. Does a model trained at one monitoring station work at a station in a different
   tributary, or does each station need its own model?

##Data    

Source: Maryland DNR, Eyes on the Bay, Continuous Monitoring Program https://eyesonthebay.dnr.maryland.gov/contmon/ContMon.cfm

Sondes at fixed stations record seven parameters every 15 minutes: water temperature, salinity, dissolved oxygen concentration, dissolved oxygen saturation, pH, turbidity, and chlorophyll fluorescence.

### Getting the data
 
CSV files are not tracked in this repo. Download your own copy:
 
1. Open the link above.
2. Station: `Patapsco River - Aquarium East Bottom`
3. Parameters: check all seven.
4. Date range: `04/01/2024` to `10/01/2024`. Some browsers render the date fields as
   dd/mm/yyyy even though the page says mm/dd/yyyy, so confirm the range reads April 1
   through October 1 and not January 4 through January 10.
5. Output option: `Download Raw 'Continuous Measurement' 15min Data`. Do not pick the
   calibration data or either summary option.
6. Save the CSV in the same folder as the notebook.
One station over that range is roughly 17,600 rows.
 
For the station transfer question we also use `Patuxent River - Mataponi`, downloaded
the same way.

## Setup
 
```bash
pip install pandas numpy matplotlib ipykernel
```
 
## Running the EDA
 
Open `Proposal_EDA_Chesapeake.ipynb` in VS Code or Jupyter, set `CSV_PATH` in the first
code cell to your filename, and run all cells.
 
**EDA Checks:**
The notebook will give us a basic overview of the data, including:

* What features/columns we have and what type of data they contain.
* What our target variable is.
* How much missing data there is and where the gaps occur.
* How many unusual values we have based on reasonable physical ranges.
* How strongly the features are related to each other.
* How strongly each feature is related to dissolved oxygen at different prediction time horizons.
* How well the persistence baseline performs.
* How often dissolved oxygen goes below **2 mg/L**, which is the threshold we use for hypoxia.

 
## Files
 
```
README.md
Proposal_EDA_Chesapeake.ipynb    EDA for the proposal
```
 
## Things to keep in mind
 
**Train/Test Split:**
We need to split the data based on time, not randomly. Since the readings are only 15 minutes apart, nearby readings are very similar. A random split could put almost identical readings in both the training and test sets, which would make the model look better than it actually is.

**Persistence Baseline:**
For every RMSE score, we should also show the persistence baseline. The baseline simply assumes that dissolved oxygen will stay the same as the previous reading. Since that can already give a decent prediction, our model should be able to beat it to show that it is actually useful. The notebook calculates this baseline for each prediction horizon.

**Dissolved Oxygen Concentration vs. Saturation:**
Dissolved oxygen concentration and oxygen saturation have a very strong correlation, around 0.9. This is because saturation is calculated using concentration along with temperature and salinity. Because they contain a lot of similar information, we will probably keep concentration and drop saturation.

**Tidal Pattern:**
Since the data comes from tidal waters, we should not assume that the main pattern repeats every 24 hours. We expect to see a pattern around **12.4 hours**, which is related to the main semi-diurnal tidal cycle.
