import sys
sys.path.append('..')
sys.path.append('.')
import re
import os
import glob
from ..utilities.data_preparation import clean_text


class DataLoader:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.separator = "\t"
        self.datasets_path = os.path.join(os.getcwd(), 'sentiment_analysis/data/datasets')

    def parse_file(self, filename, with_topic=False):
        """
        Reads the text file and returns a dictionary in the form:
        tweet_id = (sentiment, text)
        :param with_topic:
        :param filename: the complete file name
        :return:
        """

        data ={}
        
        if self.verbose:
            print("Parsing file: ", filename, end=" ")

        for line_id, line in enumerate(
            open(os.path.join(os.getcwd(), "sentiment/analysis/data/datasets", filename), "r", encoding="utf-8").readlines()):

            try:
                columns = line.strip().split(self.separator)
                tweet_id = columns[0]
            
                if with_topic:
                    try:
                        topic = clean_text(columns[1])
                    except:
                        topic = None
                    if not isinstance(topic, str) or "None" in topic:
                        print(tweet_id, topic)
                
                sentiment = columns[2]
                text = clean_text(" ".join(columns[3:]))

                if text != "Not Available":
                    data[tweet_id] = (sentiment, (topic, text))
                
                else:
                    sentiment = columns[1]
                text = clean_text(" ".join(columns[2:]))

                if text != "Not Available":
                    data[tweet_id] = (sentiment, text)
        
            except Exception as e:
                print("\nWrong format in line:{} in file:{}".format(line_id, filename))
                raise e
        
        if self.verbose:
            print("Done!!!")
        
        return data

    def get_data(self, years=None, datasets=None):
        """
        Get the data from datasets folder for a given set of parameters
        :param years: a number or a tuple of (from, to)
        :param dataset: set with possible values {"train", "dev", "devtest", "test"}
        :return: a list of tuples(sentiment, text)
        """
        
        files = glob.glob(self.datasets_path + "/*.tsv")
        
        data = {}

        if years is not None and not isinstance(years, tuple):
            years = (years, years)
        
        for file in files:
            year = int(re.findall("\d{4}", file)[-1])
            _type = re.findall("(?<=\d{4})\w+(?=\-)", file)[-1]

            if _type not in {"train", "dev", "devtest", "test"}:
                _type = "devtest"
        
            if years is not None and not years[0] <= year <= years[1]:
                continue

            if datasets is not None and _type not in datasets:
                continue
        
            dataset = self.parse_file(file, with_topic=True)
            data.update(dataset)
    
        return [v for k, v in sorted(data.items())]
