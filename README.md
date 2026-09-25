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

## Data

Source: Maryland DNR, Eyes on the Bay, Continuous Monitoring Program
https://eyesonthebay.dnr.maryland.gov/contmon/ContMon.cfm

Sondes at fixed stations record seven parameters every 15 minutes: water temperature,
salinity, dissolved oxygen concentration, dissolved oxygen saturation, pH, turbidity,
and chlorophyll fluorescence.

The Aquarium East Bottom file is committed as `EDA/Test_Station.csv`, so you do not
need to download it to run the notebook. The steps below are how it was obtained, and
how to get the second station for research question 2.

### Getting the data

1. Open the link above.
2. Station: `Patapsco River - Aquarium East Bottom`, or `Patuxent River - Mataponi`
   for the station transfer question.
3. Parameters: check all seven. Blue Green Algae is an eighth checkbox at some
   stations, but it comes through empty here and the notebook drops it automatically.
4. Date range: `04/01/2024` to `10/01/2024`. Some browsers render the date fields as
   dd/mm/yyyy even though the page says mm/dd/yyyy, so confirm the range reads April 1
   through October 1 and not January 4 through January 10.
5. Output option: `Download Raw 'Continuous Measurement' 15min Data`. Do not pick the
   calibration data or either summary option.
6. Save the CSV in the `EDA` folder and set `CSV_PATH` in the notebook's first code
   cell to match the filename.

One station over that range is 17,664 rows. After missing values, 13,689 of those have
a usable dissolved oxygen reading.

## Setup

```bash
pip install pandas numpy matplotlib ipykernel
```

## Running the EDA

Open `EDA/Proposal_EDA_Chesapeake.ipynb` in VS Code or Jupyter, set `CSV_PATH` in the
first code cell to your filename, and run all cells.

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

The printed output is also saved to `EDA/eda_numbers_for_writeup.txt` so numbers can be
copied into the write-up without re-running anything.

## Files

```
README.md
.gitignore
.gitattributes
EDA/
├── Proposal_EDA_Chesapeake.ipynb    EDA for the proposal
├── Test_Station.csv                 Aquarium East Bottom, Apr 1 to Oct 1 2024
└── eda_numbers_for_writeup.txt      printed output from the notebook
```

## Things to keep in mind

**Train/Test Split:**
We need to split the data based on time, not randomly. Since the readings are only 15 minutes apart, nearby readings are very similar. A random split could put almost identical readings in both the training and test sets, which would make the model look better than it actually is.

**Do Not Delete Rows With Missing Values:**
This is time series data and every row sits exactly 15 minutes after the one before it. The forecast horizons are calculated by shifting a fixed number of rows, so 24 rows means 6 hours. If we delete rows, that stops being true and "6 hours ahead" quietly becomes something else. The code still runs and nothing warns us. Keep the rows, leave the gaps as NaN, and drop incomplete cases only after the shifted target column has been built.

**Negative Dissolved Oxygen Readings Are Real:**
The physical range screen uses a negative lower bound for dissolved oxygen and saturation on purpose. A sonde sitting in water with no oxygen left reads slightly below zero because of calibration offset. At this station those readings run from about -0.12 to -0.01 mg/L, and there are 2,769 of them, which is roughly 40% of all our hypoxic readings. Screening them out at zero would delete the phenomenon we are trying to predict. The notebook screens wide enough to catch genuinely broken values and clips the small negatives up to zero instead.

**Persistence Baseline:**
For every RMSE score, we should also show the persistence baseline. The baseline simply assumes that dissolved oxygen will stay the same as the current reading. Since that can already give a decent prediction, our model should be able to beat it to show that it is actually useful. The notebook calculates this baseline for each prediction horizon. For Aquarium East Bottom it currently comes out as:

| Horizon | RMSE (mg/L) | MAE (mg/L) | 20% improvement target |
| --- | --- | --- | --- |
| 1 hour | 0.581 | 0.370 | 0.465 |
| 3 hours | 0.864 | 0.558 | 0.691 |
| 6 hours | 1.163 | 0.746 | 0.931 |
| 12 hours | 1.592 | 1.025 | 1.273 |
| 24 hours | 2.248 | 1.479 | 1.798 |

**Dissolved Oxygen Concentration vs. Saturation:**
Dissolved oxygen concentration and oxygen saturation correlate at 0.987. This is because saturation is calculated using concentration along with temperature and salinity, so they are not two separate measurements. We will probably keep concentration and drop saturation. Saturation also correlates with pH at 0.839, which is the second collinear pair we need to resolve.

**Periodicity:**
Do not assume a daily cycle at this station. We measured the autocorrelation of dissolved oxygen at Aquarium East Bottom and there are no peaks at all in the first 30 hours. It decays smoothly, 0.929 at 6 hours, 0.856 at 12.4 hours, 0.720 at 24 hours. This is bottom water at 6.6 m sitting under a pycnocline, so it is cut off from both tidal mixing and surface photosynthesis. It drifts toward anoxia and gets reset by occasional mixing events rather than cycling.

A 12.4 hour tidal signal does show up at shallower stations such as Patuxent River Iron Pot Landing, so this may change if we add a second station. For Aquarium East Bottom it means an hour-of-day or tidal-phase baseline would add almost nothing, and persistence is the baseline that matters.
