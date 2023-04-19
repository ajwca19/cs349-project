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
    grouped_reviews = review_df.groupby('asin')
    
    asin_review_data_dict = {}
    for ind in product_df.index:
        review_group = grouped_reviews.get_group(review_df['asin'][ind])
        asin_review_data_dict.update({product_df['asin'][ind] : review_group})
        
    return asin_review_data_dict
    
if __name__ == '__main__':
    # dictionary of ASINS and the reviews for the ASIN
    asin_review_data_train = data_preprocessing("../devided_dataset_v2/CDs_and_Vinyl/train", False)

# list of ASINS (keys for the dictionary)
asins = list(asin_review_data_train.keys())