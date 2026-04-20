# Notebook 11
## Sorting Methods
np.sort(arr)
- creates a new sorted copy of the array
- return sorted array
arr.sort()
- modifies original array
- returns nothing
## Navigating Multi-dimensional Sorting
axis = 1
- sorts horizontally
axis = 0
- sorts vertically
axis = -1
- sorts along the last axis
EX: np.sort(A,axis=0)
## Reverse Sorting & Slicing
Slicing
- [::1] is used after a sort to flip the order. 
- EX: np.sort(arr)[::-1] yields a descending list.
np.flip()
- more explicit function that reverses the order of elements along given axis.
## argsort Logic
doesn't return the sorted values, returns the indicies that would put the array in order
- EX: grades = [94,87,96] then np.argsort(grades) returns [1,0,2]
primarily used to sort "parallel arrays"


# Notebook 12
## Anatomy of Pandas Objects
Series
- One dimensional labeled array
Data Frame
- Two dimensional, size-mutable, tabular data structure with labeled axes.
Metadata Attributes
    .index
    - returns the row labels
    .columns
    - returns names of columns
    .values
    - returns data as a numpy array, w/o labels
## Selection Logic
.loc[label]
- uses name of index or column
    - EX: df.loc[0] looks for the row where the label is 0.
.iloc[position]
- uses zero-based physical position
    - df.iloc[0] looks for the first row, regardless of what its label is.
If you have an index that has been shuffled, df.loc[0] will give you the second row, but df.iloc[0] will give you the first.
## Vectorization
S + 5 adds 5 to every single value in Series S
S ** 2 squares every value
Filtering (df[['Grades'] > 85])
- Mask
    - df['Grades'].... creates a Series of True or False
- Filter
    - outer df[.....] only keeps the rows where the mask is true

# Notebook 13
## Handling Missing Data
df.dropna()
- removes any row that contains at least one missing vlaue (NaN)
df.fillna(value)
- replaces all NaN instances with a specific value
Like sorting, these functions typically return a new DataFrame
- Unless inplace=true is used, the original df still contains the missing data in the next line of code
## Functional Transformation
.apply() is used to pass each element of a column through a function.
- You can pass a pre-defined Python function:
    - df['Score'].apply(my_function)
- Lambda Functions
    - EX: df['Grade'].apply(lambda x: x + 5)
## Split-Apply-Combine Strategy
df.groupby('Category')['Value'].mean()
- Split -> Data is partitioned into groups
- Apply -> Mathematical function is calculated for each group individually
- Combine -> Results are put back together into a new series or dataframe where category is the new index
## Sorting DataFrames
df.sort_values()
- by='ColumnName' -> must specify which column drives the sort
- ascending=False -> sort in descending order

# Notebook 14
## Three Primary Plot Types
plt.plot(x,y) [Line Plot]
- Connects data points with lines. Used to show trends or functional relationships
plt.scatter(x,y) [Scatter Plot]
 - Draws individual dots without connecting lines.
 - Primarily used to show correlations or clusters in data.
plt.bar(categories, values) [Bar Chart]
- Used for discrete, categorical data
## Plot Anatomy
plt.xlabel() / plt.ylabel()
- sets the titles for the horizontal and vertical axes.
plt.title()
- sets the main heading at the top of the plot.
plt.show()
- command required to render the plot to the screen.
## Layout Logic
Logic plt.subplot(1,2,1)
- 1: There is 1 row of plots
- 2: There are 2 columns of plots
- 1: This code refers to the first plot
Indices
- Subplot indices start at 1, not 0.
## Statistical Plots
Histograms (plt.hist)
- Purpose -> Show frequency distribution of a single dataset.
- Mechanism -> Groups data into "bins". Height of the bar represents how many data points fall into that specific range.
Boxplots (plt.boxplot)
- Purpose -> Summarizes the "Five Number Summary" of a dataset
- Theoretical Components
    - Box -> Inter Quartile Range (middle 50% of the data)
    - Line in Box -> Median of data
    - Whiskers -> Extend to the minimum and maximum values
    - Circles/Dots -> Individual points plotted beyond the whiskers are outliers

# Notebook 15
## Exploratory Data Analysis
df.info()
- Shows the data type of each column and the number of non-null entries.
df.describe()
- Provides a statistical summary of numerical columns, including the mean, standard deviation, and the Five-Number Summary
df.value_counts()
- counts the frequency of unique entries in categorical column
df.corr()
- correlation matrix
- Logic -> Values range from -1 to 1
- 1 is perfect positive correlation
- 0 is no relationship
- -1 is perfect negative correlation
## Feature Engineering
One-Hot Encoding
- pd.get_dummies -> converts a categorical column into multiple binary columns where 1 means present and 0 absent
Dropping Column
- df.drop -> Used to remove "noisy" data or the target variable itself from the features set (X)
The Split (X and Y)
- X (Features) -> Independent variables used to make predictions
- Y (Target) -> Specific value the model is trying to learn to predict
## Machine Learning Workflow
Train-Test Split
- Function -> train_test_split(X, y, test_size=0.2)
- Theory -> We split data so the model "learns" on the training set and we "evaluate" it on a testing set it has never seen before.
## Linear Regression
Evaluation Math
- Mean Squared Error -> Average of the squares of the errors.
    - (1/n) (yi - y-hati)^2
    - Smaller values are better because they indicate the predictions are closer to actual values
- R-Squared -> Coefficient of Determination
    - Represents the percentage of variance in the target variable that the model explains.
    - Values -> Closer to 1.0 indicates a perfect fit; closer to 0 indicates the model is no better than just guessing the average

# Notebook 17
## 3D Array Structure
Shape
- (Height, Width, Channels)
Channels
- (R,G,B)
Data Type
- often use uint8 (pixel range from 0 to 255)
## Slicing and Cropping
Spatial Slicing
- img[0:100, 0:200, :]
Channel Slicing
- img[:, :, 0]
## Theoretical Image Transformations
Brightness and Color Shifting
- Addition/Subtracting
    - Adding a scalar increases the intensity of pixels, making image brighter
- Selective Shifting
    - img[:, :, 0] = 255 would turn every pixel's red component to maximum, giving the image a heavy red tint.
## Axis Logic
img.min(axis=2)
- axis=0: Operations along the height (rows).
- axis=1: Operations along the width (columns).
- axis=2: Operations through the color channels (depth).

# Notebook 18
## Bulk Math and axis Parameter
Horizontal Summation: df.iloc[:, 4:10].sum(axis=1)
- iloc[:, 4:10] -> Selects all rows and columns from index 4 up to (but not including) 10.
- axis=1 -> Tells Pandas to sum across the columns for each row, rather than down the rows.
Creating New Columns
- You can assign the result of these bulk operations directly to a new column name, such as df['Total'].