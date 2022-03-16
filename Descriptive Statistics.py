
"""Statistics is divided into two major areas:

Descriptive statistics: describe and summarize data;

Inferential statistics: methods for using sample data to make general 
conclusions (inferences) about populations. 

Descriptive statistics of both numerical and categorical variables and 
is divided into two parts:

Measures of central tendency;
Measures of spread.

"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
path="C:\\Users\Admin\Downloads\house_prices.csv"
df = pd.read_csv(path)
df

# Print the top rows
df.head()

# Before stats, a quick look at the data
df.info()

# Review the data types of attributes in your data.
df.dtypes

# Summarize the distribution of instances across classes in your dataset.
df.describe()


saleprice = df['SalePrice']

#mean
mean=saleprice.mean()

mean

#median
median=saleprice.median()

median

#mode
mode=saleprice.mode()

mode

#printing mean, median and mode
print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode)

#visualizing through histogram

plt.figure(figsize=(10,5)) #width and height
plt.hist(saleprice,bins=100,color='orange')
plt.axvline(mean,color='red',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='green',label='Mode')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.show()


saleprice.min() #maximum value of salePrice

saleprice.max() #minimum value of salePrice

#Range
saleprice.max()-saleprice.min()

#variance
saleprice.var()

from math import sqrt

#standard deviation
std = sqrt(saleprice.var())
std

saleprice.std()

"""Skewness
In a perfect world, the data’s distribution assumes the form of a 
bell curve (Gaussian or normally distributed), but in the real world, 
data distributions usually are not symmetric (= skewed).
Therefore, the skewness indicates how much our distribution derives 
from the normal distribution (with the skewness value of zero or 
                              very close).

There are three generic types of distributions:
Symmetrical [median = mean]: In a normal distribution, 
the mean (average) divides the data symmetrically at the median value 
or close.

Positive skew [median < mean]: The distribution is asymmetrical, 
the tail is skewed/longer towards the right-hand side of the curve. 
In this type, the majority of the observations are concentrated on 
the left tail, and the value of skewness is positive.

Negative skew [median > mean]: The distribution is asymmetrical and 
the tail is skewed/longer towards the left-hand side of the curve. 
In this type of distribution, the majority of the observations 
are concentrated on the right tail, and the value of skewness 
is negative.

Rule of thumbs:
Symmetric distribution: values between -0.5 to 0.5.
Moderate skew: values between -1 and -0.5 and 0.5 and 1.
High skew: values <-1 or >1.

"""
saleprice.skew()


"""
Kurtosis
kurtosis is another useful tool when it comes to quantify the shape 
of a distribution. It measures both how long are the tails, 
but most important, and how sharp is the peak of the distributions.
If the distribution has a sharper and taller peak and 
shorter tails, then it has a higher kurtosis while a low kurtosis 
can be observed when the peak of the distribution is flatter with 
thinner tails. 

There are three types of kurtosis:

Leptokurtic: The distribution is tall and thin. 
The value of a leptokurtic must be > 3.

Mesokurtic: This distribution looks the same or very similar to a 
normal distribution. The value of a “normal” mesokurtic is = 3.

Platykurtic: The distributions have a flatter 
and wider peak and thinner tails, meaning that 
the data is moderately spread out. 

The value of a platykurtic must be < 3.

"""
#kutosis
"""
kurtosis is another useful tool when it comes to quantify 
the shape of a distribution. 
It measures both how long are the tails, but most important, 
and how sharp is the peak of the distributions.
If the distribution has a sharper and taller peak and 
shorter tails, then it has a higher kurtosis 
while a low kurtosis can be observed when the peak of the 
distribution is flatter with thinner tails. 

There are three types of kurtosis:
    
Leptokurtic: The distribution is tall and thin. 
The value of a leptokurtic must be > 3.

Mesokurtic: This distribution looks the same or very similar 
to a normal distribution. The value of a “normal” mesokurtic is = 3.

Platykurtic: The distributions have a flatter and wider peak a
nd thinner tails, meaning that the data is moderately spread out. 
The value of a platykurtic must be < 3.

"""
saleprice.kurt()

#Boxplot
plt.boxplot(saleprice)
plt.show()