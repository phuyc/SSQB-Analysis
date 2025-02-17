import pandas as pd
from IPython.display import display
import json

with open('new_practice_tests.json', 'r', encoding="utf8")as file:
    # load as dataframe using json_normalize method
    df = pd.read_json(file)
    display(df)