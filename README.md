# Urban Boundary Layer Observations
This repo holds scripts used to collect and process data to support the observation of the atmospheric/planetary boundary layer over the New York City metropolitan area. This work is a project led by the Urban Flux Observatory at the City College of New York and is funded by Department of Defense Army Research Office Grant #W911NF2020126.

## Observation Locations
| Abbreviation  | Borough       | Location                  | Coordinates         |
| ------------- | ------------- | ------------------------- | ------------------- |
| BKLN          | Brooklyn      | Brooklyn College          | 40.6311, -73.9524   |
| BRON          | Bronx         | Lehman College            | 40.8729, -73.8945   |
| MANH          | Manhattan     | City College              | 40.8200, -73.9493   |
| QUEE          | Queens        | Queens College            | 40.7282, -73.7949   |
| STAT          | Staten Island | College of Staten Island  | 40.6021, -74.1504   |


## Data Sources
### Flux Towers
* BKLN (NYS Mesonet)
* MANH (NOAA-CESSRST)
* QUEE (NYS Mesonet)
* STAT (NYS Mesonet)

### NYS Mesonet Lidar Stations
* BRON (Lehman College, The Bronx)
* CCNY (City College of New York, Manhattan)
* QUEE (Queens College, Queens)
* STAT (College of Staten Island, Staten Island)
* WANT (Cedar Creek Wastewater Treatment Plant, Wantagh)

## Sample Figures
Adding these figures as a rough visualization of the works in progress. Much more polished figures will be used for the upcoming manuscript.

![Quiver plot at the College of Staten Island with vertical velocity overlaid. Note that this is a mean profile over days with high probabilities of sea-breeze observation (winds <1 km in height possessing mean horizontal speeds of <5 m/s). Note the southerly shift in wind direction from 12:00 to 19:00 after strong updrafts, with a mid-level northerly shif and stronger downdrafts. This indicates a mean sea-breeze circulation in the absence of geostrophic winds.](https://github.com/mr-gabrielrios/urban_boundary_layer_obs/blob/main/dev/figs/sea-breeze_staten-island.png)

![Normalized power spectra over the City College flux tower in upper Manhattan at 3 heights in the mixed layer. Normalization of the spectra occurs using frequency and mean horizontal wind speed.](https://github.com/mr-gabrielrios/urban_boundary_layer_obs/blob/main/dev/figs/spectra-y_U_norm.png) 
