import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import load_data

def test_load_data():
    df = load_data()
    assert df is not None
    assert len(df) > 0
    assert 'default' in df.columns

def test_data_shape():
    df = load_data()
    assert df.shape[0] == 10000
    assert df.shape[1] == 8

def test_default_rate():
    df = load_data()
    default_rate = df['default'].mean()
    assert 0 < default_rate < 1