# Spotting the Curves in the US
**A Better Overview of COVID-19 in Each State of the United States**

<img src="https://github.com/streamlit/streamlit/raw/develop/docs/_static/img/simple_example.png"/>

## Purpose and Features
To better understand COVID-19 in the US, this dashboard presents an overview of state-level data with complementary information. This dashboard aims to show individual trends within each state, extract timely state-level phase information and **provide a consistent and easy-to-read platform other than media outlets**.

## How to run this
The demo requires Python 3.6+. I would suggest creating a new virtual environment, then running:

```sh
git clone https://github.com/HiIamJeff/covid19-curves-in-US
cd covid19-curves-in-US
pip install -r requirements.txt
python run temp.py
```


## References and Relevant Readings
The state-level data is from [The New York Times GitHub](https://github.com/nytimes/covid-19-data). 
The information about the phases of each state is retrieved from 
[The New York Times](https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html).
Here is another [country-level dashboard](https://spot-the-curve-coronavirus.herokuapp.com/) and 
a [Medium article](https://towardsdatascience.com/spot-the-curve-visualization-of-cases-data-on-coronavirus-8ec7cc1968d1?source=friends_link&sk=4f984ca1c1e4df9535b33d9ccab738ee) 
about interpreting Coronavirus data. 

## Questions? Comments?
If you notice any mistakes or have any comments regarding the data, visualizations or dashboard, please feel free to contact the author (*[Jeff Lu](https://www.linkedin.com/in/jefflu-chia-ching-lu/)*)



## Main Features
Here are just a few of the things that pandas does well:
  - Powerful, flexible [**group by**][groupby] functionality to perform
    split-apply-combine operations on data sets, for both aggregating
    and transforming data

   [missing-data]: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
   [insertion-deletion]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#column-selection-addition-deletion
   [alignment]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html?highlight=alignment#intro-to-data-structures
   [groupby]: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#group-by-split-apply-combine
   [conversion]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/pandas-dev/pandas

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/pandas) and on [Conda](https://docs.conda.io/en/latest/).



