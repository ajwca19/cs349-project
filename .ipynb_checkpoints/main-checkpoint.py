import pandas as pd

class Product:
    def init(self, asin, reviews, awesome):
        self.asin = asin
        self.reviews = reviews
        self.awesome = awesome

#take in a directory path (as string), return a list of Product objects organized by unique asin
def data_preprocessing(path, test=False):
    #create appropriate file path
    if test == False:
        pfilename = path + "/product_training.json"
        rfilename = path + "/review_training.json"
    else:
        pfilename = path + "/product_test.json"
        rfilename = path + "/review_test.json"
    
    #extract files as pandas dataframes
    product_df = pd.read_json(pfilename)
    review_df = pd.read_json(rfilename)
    
    
    
if __name__ == '__main__':
    data_preprocessing("devided_dataset_v2/Grocery_and_Gourmet_Food/test1", True)