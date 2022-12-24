# Australia Fire dataset:

Can  be downloaded from here: https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland

fire nrt V1 96617.csv which is described here: https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/viirs-i-band-375-m-active-fire-data

### Plot the longitude vs latitude several ways:
1. Using the matplotlib defaults
2. Adjusting alpha and marker size to compensate for overplotting
3. Using a hexbin plot
4. Subsampling the dataset

### Visualizing class membership

Visualize the distribution of Brightness temperature I-4 as a histogram (with appropriate settings). Let’s assume we are certain of a fire if the value of temperature I-4 is saturated as visible from the histogram.

1. Small multiple plots of whether the brightness is saturated, i.e. one plot of lat vs long for those points with brightness saturated and a separate for those who are not
2. Plot both groups in the same axes with different colors. Try changing the order of plotting the two classes (i.e. draw the saturated first then the non-saturated or the other way around)
