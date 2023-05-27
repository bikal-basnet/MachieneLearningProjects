# Product Delivery Time Structurer
Detect the best structure for the product delivery service to meet the needs of each customer.

#Description: 
In this project, we use the  customers annual spending amount across various  product categories such as Fresh Products, Milk Products, Grocery, Frozen  and Detergent and predict the best delivery structure .
The data will be used to describe the variation in different types of customers and ultimately will be used to gauge, if changing the delivery cycle from  existing 3 days to 5 days will impact the customer. And if so , in what ways  is it likely to do so, positiely or negatively.

We use  Principal Component Analysis to  reduce the dimensionality of the data, along with the dta preprocssing  and then use K-means cluster over the the Gaussian Mixture  Clustering models to  identify customer clusters. 
The customer clusters are then used as the basis of the A/B experiments, the plan for which has also been layed out in the project.

Since we A/B experiment performance is out of scope of the project, we hypothesize based on the cluster defination, the potential impact of the delivery schedule change and propose a  plan for the delivery structure.


## Getting Started Guide:

### Install

Step 1: This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Step 2 : Also, you need to install ipython and  jupyter notebook as
```
pip install ipython
pip install jupyter
```

Step 3: Go to the directory containing the project then open the notebook as
```
jupyter notebook
```

Step 4:  Then jupyter notebook will open in the browser. 

Step 5: Traverse to the ".ipynb" file and you can then change or run the program as desired in the browser notebook itself.

## Project Structure
- "customers.csv" : The data  containing the customer attributes. You can find more information on this dataset on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) page. 
- "submissions_bikal_basnet/customer_segments.html" : This is the "html" version of the project and can be directly viewed in the browser.
- "customer_segments.ipynb" : This is the  ipython notebook file of the project. If you want to tweak it, make changes and play around, then you can open the ipython notebook file in the jupyter notebook and run it directly in your browser, as you go on making the changes.
- "renders.py" : Helper python file containing  helper functions such as creating a dataframe of Principal Componenet Analysis results, visualising the PCA- reduced clusters e.t.c

