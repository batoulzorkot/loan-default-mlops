import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

def test_load_data():
    path = 'data/Loan_Data.csv'
    if not os.path.exists(path):
        path = '../data/Loan_Data.csv'
    df = pd.read_csv(path)
    assert df is not None
    assert len(df) > 0
    assert 'default' in df.columns

def test_data_shape():
    path = 'data/Loan_Data.csv'
    if not os.path.exists(path):
        path = '../data/Loan_Data.csv'
    df = pd.read_csv(path)
    assert df.shape[0] == 10000
    assert df.shape[1] == 8

def test_default_rate():
    path = 'data/Loan_Data.csv'
    if not os.path.exists(path):
        path = '../data/Loan_Data.csv'
    df = pd.read_csv(path)
    default_rate = df['default'].mean()
    assert 0 < default_rate < 1