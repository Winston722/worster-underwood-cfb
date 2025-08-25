from __future__ import annotations

import time
import cfbd
from pprint import pprint
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from cfbd.models.division_classification import DivisionClassification
from cfbd.models.game import Game
from cfbd.models.season_type import SeasonType
from cfbd.configuration import Configuration
from cfbd.api_client import ApiClient
from cfbd import GamesApi # type: ignore



def get_college_football_games(year=2024):
    """
    Fetch college football games data from CFBD API for a given year.
    
    Args:
        year (int): The year to fetch games for. Defaults to 2024.
    
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - df (pandas.DataFrame): Current year's Division I games (FBS + FCS) 
                                   with duplicates removed based on game ID
            - ly_df (pandas.DataFrame): Previous year's Division I games (FBS + FCS) 
                                      with duplicates removed based on game ID
    
    Raises:
        Exception: If API call fails or other errors occur during data fetching.
    """
    load_dotenv()
    
    # Defining the host is optional and defaults to https://api.collegefootballdata.com    
    # Configure Bearer authorization: apiKey
    configuration = Configuration(
        host = "https://api.collegefootballdata.com",
        access_token = os.environ.get("CFBD_API_KEY")
)
    
    with ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = GamesApi(api_client)
        
        try:
            fbs = api_instance.get_games(year=year, classification=DivisionClassification('fbs'))
            fcs = api_instance.get_games(year=year, classification=DivisionClassification('fcs'))
            ly_fbs = api_instance.get_games(year=year-1, classification=DivisionClassification('fbs'))
            ly_fcs = api_instance.get_games(year=year-1, classification=DivisionClassification('fcs'))
            all_d1 = fbs + fcs
            df = pd.DataFrame([g.to_dict() for g in all_d1]).drop_duplicates(subset=["id"])
            ly_all_d1 = ly_fbs + ly_fcs
            ly_df = pd.DataFrame([g.to_dict() for g in ly_all_d1]).drop_duplicates(subset=["id"])
            return df, ly_df
        except Exception as e:
            print("Exception when calling GamesApi->get_games: %s\n" % e)
            raise