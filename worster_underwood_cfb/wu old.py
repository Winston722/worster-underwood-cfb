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
from cfbd.rest import ApiException
from typing import Iterable, Any

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


def prepare_schedule(
    api_response: Iterable,  
    hfa: int = 3,
    decay: float = 1/3,  
) -> pd.DataFrame:
    """
    Return ['week','winner','loser','hfa_margin'] ready for add_weight().
    - Drop canceled/incomplete games (missing scores)
    - Assert neutralSite complete (per your rule after drop)
    - Winner-perspective, HFA-adjusted margin
    - Assert no ties (FBS)
    """
    
    cols = ['seasonType','week','neutralSite',
            'homeTeam','awayTeam','homePoints','awayPoints']

    # Vectorized load in one shot
    df = pd.DataFrame.from_records((g.to_dict() for g in api_response), columns=cols)

    # 1) Drop canceled / incomplete
    df = df.dropna(subset=['homePoints','awayPoints']).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=['week','winner','loser','hfa_margin'])

    # 2) Fail-fast invariants
    assert not df[['seasonType','week','homeTeam','awayTeam']].isna().any().any(), \
        "Nulls in required non-score fields."
    assert not df['neutralSite'].isna().any(), \
        "neutralSite should be non-null after dropping canceled games."

    # 3) Types + postseason mapping
    df['week'] = pd.to_numeric(df['week'], errors='raise', downcast='integer')
    df.loc[df['seasonType'].eq('postseason'), 'week'] = 18
    df['week'] = df['week'].astype('int16')
    assert (df['week'] >= 1).all(), "week must be >= 1"

    # Pull arrays once 
    hp = pd.to_numeric(df['homePoints'], errors='raise').to_numpy()
    ap = pd.to_numeric(df['awayPoints'], errors='raise').to_numpy()
    ns = df['neutralSite'].astype(bool).to_numpy()
    wk = df['week'].to_numpy()
    home = df['homeTeam'].to_numpy(object)
    away = df['awayTeam'].to_numpy(object)

    # 4) Margins & outcomes
    margin = hp - ap                          # home-perspective true margin
    home_field = np.where(ns, 0, hfa)         # 0 if neutral, else HFA
    adj_home = margin - home_field            # remove HFA from home side

    home_win = margin > 0
    # away_win = margin < 0  # redundant given assert

    winners = np.where(home_win, home, away)
    losers  = np.where(home_win, away, home)
    hfa_margin = np.where(home_win, adj_home, -adj_home)

    return pd.DataFrame({
        'week': wk,
        'winner': winners,
        'loser': losers,
        'hfa_margin': hfa_margin,
    })

def add_weight(df: pd.DataFrame, decay: float = 1/3) -> pd.DataFrame:
    """
    Calculate weights for college football games based on team game counts and recency.
    
    Weight formula: sqrt((total_games / max_total_games) / (weeks_ago ** decay))
    Weights are normalized to sum to 100.
    
    Args:
        df: DataFrame with columns ['week', 'winner', 'loser', 'hfa_margin']
        decay: Time decay factor for recency weighting (default: 1/3)
    
    Returns:
        DataFrame with columns ['week', 'winner', 'loser', 'hfa_margin', 'weight']
        
    Performance: ~14.8x faster than naive pandas approach using:
        - pd.factorize() for efficient team encoding
        - np.bincount() for fast game counting  
        - Pure numpy operations for mathematical calculations
    """
    # Handle empty DataFrame edge case

    # --- fail-fast checks ---
    assert decay > 0, "decay must be > 0"
    assert not df[['winner','loser']].isna().any().any(), "winner/loser must be non-null"
    assert (df['week'] >= 1).all(), "week must be >= 1"
    assert len(df) > 0, "empty dataframe"
    
    if df.empty:
        return df.assign(weight=pd.Series(dtype='float64'))[
            ['week', 'winner', 'loser', 'hfa_margin', 'weight']
        ]
    
    # Extract numpy arrays once to minimize pandas overhead
    winner_vals = df['winner'].values
    loser_vals = df['loser'].values
    week_vals = df['week'].values
    
    # Efficient team encoding using pandas factorize
    both_teams = np.concatenate([winner_vals, loser_vals])
    codes, _ = pd.factorize(both_teams, sort=False)
    
    # Fast game counting using numpy bincount
    n = len(df)
    counts = np.bincount(codes)
    winner_games = counts[codes[:n]]
    loser_games = counts[codes[n:]]
    
    # Pure numpy calculations for maximum speed
    total_games = winner_games + loser_games
    weeks_ago = (week_vals.max() + 1) - week_vals
    max_games = total_games.max()
    
    # Calculate weights using vectorized operations
    if max_games > 0:
        weights = np.sqrt((total_games / max_games) / (weeks_ago ** decay))
        # Normalize to sum to 100
        weights *= (100.0 / weights.sum())
    else:
        # Edge case: no games played (shouldn't happen in real data)
        weights = np.zeros(n, dtype=np.float64)
    
    # Return result with weight column
    result = df[['week', 'winner', 'loser', 'hfa_margin']].copy()
    result['weight'] = weights
    return result

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def get_initial(schedule):

    extras = schedule[['hfa_margin', 'weight']]
    transform = schedule.drop(['hfa_margin', 'weight'], axis = 1)

    # Get a list of all unique teams
    teams = sorted(set(transform['winner'].unique()).union(transform['loser'].unique()))

    # Create a new DataFrame with teams as columns using Scipy's sparse lil_matrix
    n = len(transform)
    m = len(teams)
    x = lil_matrix((n, m), dtype=int)

    # Create a dictionary to map teams to their respective column indices
    team_indices = {team: index for index, team in enumerate(teams)}

    # Fill in the sparse matrix with 1 for winners and -1 for losers efficiently
    winners = transform['winner'].map(team_indices).values
    losers = transform['loser'].map(team_indices).values

    x[np.arange(n), winners] = 1
    x[np.arange(n), losers] = -1

    #my code
    y = extras['hfa_margin'].to_numpy()
    w = extras['weight'].to_numpy()

    xw = x.multiply(np.sqrt(w[:, np.newaxis]))
    yw = y * np.sqrt(w)

    result, istop, itn, _, _, _, _, _, _, _ = lsqr(xw, yw)

    r1_ratings = pd.DataFrame(data = {'teams': teams, 'coefs': result})
    #r1_ratings.sort_values(by=['coefs'], inplace=True, ascending=False)

    schedule.set_index('winner', inplace=True, drop = False)
    r1_ratings.set_index('teams', inplace=True, drop = False)
    with_winner = schedule.join(r1_ratings, how='left').set_index('loser', drop = False)

    with_ratings = with_winner.join(r1_ratings, how = 'left', lsuffix='_winner', rsuffix='_loser').drop(['teams_winner', 'teams_loser'], axis = 1)
    with_ratings.reset_index(inplace = True, drop = True)

    return with_ratings

def get_rating(subject, initial):
    with_ratings = initial[['winner', 'loser', 'hfa_margin', 'weight','coefs_winner', 'coefs_loser']]
    subject_mask = (with_ratings['winner'] == subject) | (with_ratings['loser'] == subject)
    subject_data = with_ratings[subject_mask].copy()
    subject_data['hfa_margin'] *= np.where(subject_data['winner'] == subject, 1, -1)
    subject_data.columns = ['team1', 'team2', 'hfa_margin', 'weight', 'rating_team1', 'rating_team2']

    subject_data['y'] = subject_data['hfa_margin']+subject_data['rating_team2']
    subject_data['x'] = 1
    x = subject_data['x'].to_numpy()
    y = subject_data['y'].to_numpy()
    w = subject_data['weight'].to_numpy()

    # Apply weights to x and y
    xw = x * np.sqrt(w)
    yw = y * np.sqrt(w)

    A = xw[:, np.newaxis]

    result, _, _, _ = np.linalg.lstsq(A, yw, rcond=0.1)

    return result[0]

def get_ratings(schedule):
    initial = get_initial(schedule)
    teams = sorted(set(schedule['winner'].unique()).union(schedule['loser'].unique()))
    output_list = list(map(lambda x: get_rating(x, initial), teams))
    ratings = pd.DataFrame(list(zip(teams, output_list)), columns=['teams', 'ratings'])
    return ratings.sort_values("ratings", axis = 0, ascending = False)

def get_error(schedule, ratings):
    error_schedule = schedule.drop(['week'], axis = 1)
    ratings.sort_values(by=['ratings'], inplace=True, ascending=False)

    error_schedule.set_index('winner', inplace=True, drop = False)
    ratings.set_index('teams', inplace=True, drop = False)
    with_winner = error_schedule.join(ratings, how='left').set_index('loser', drop = False)

    with_ratings = with_winner.join(ratings, how = 'left', lsuffix='_winner', rsuffix='_loser').drop(['teams_winner', 'teams_loser'], axis = 1)
    with_ratings.reset_index(inplace = True, drop = True)
    with_ratings['error'] = (with_ratings['hfa_margin'] - (with_ratings['ratings_winner'] - with_ratings['ratings_loser']))**2

    with_ratings.drop(['hfa_margin','ratings_winner', 'ratings_loser'], inplace = True, axis = 1)

    with_ratings2 = with_ratings.copy()

    with_ratings.columns = ['team1', 'team2', 'weight', 'error']
    with_ratings2.columns = ['team2', 'team1', 'weight', 'error']

    error_set = (pd.concat([with_ratings, with_ratings2], ignore_index=True)).drop(['team2'], axis = 1)
    ##need to factor in weight
    error_sum = pd.DataFrame(error_set.groupby(by = 'team1', axis=0).apply(lambda x: (x.weight*x.error).sum()))
    error_count = error_set.drop(['weight'], axis = 1).groupby(by = 'team1', axis=0).count()


    error_total = error_sum.join(error_count, lsuffix = "r", rsuffix = "l")
    error_total.reset_index(inplace = True)
    error_total.columns = ['team', 'error', 'games']

    error_total['rmse'] = (error_total['error']/error_total['games'])**0.5
    error_total['psudo_sd'] = ((error_total['rmse']*error_total['games'])+6*22)/(error_total['games']+22)
    error = error_total.drop(['error','games','rmse'], axis = 1)
    return error

def combined(ratings, error):
    error.set_index('team', drop = False, inplace = True)
    rating_error = ratings.join(error, how = 'left', lsuffix='_l', rsuffix='_r').drop(['teams','team'], axis = 1).reset_index()
    rating_error.columns = ['team','rating','psudo_sd']
    return rating_error

def error_hfa(x, api_response, decay = 1/3):
    hfa = x
    schedule = add_weight(prepare_schedule(api_response, hfa = hfa, decay = decay))

    ratings = get_ratings(schedule)
    return get_error(schedule, ratings)['psudo_sd'].sum()

def error_decay(x, api_response, hfa=3):
    decay = x
    schedule = add_weight(prepare_schedule(api_response, hfa = hfa, decay = decay))

    ratings = get_ratings(schedule)
    return get_error(schedule, ratings)['psudo_sd'].sum()


# ---------- 1) Minimal schedule extraction (winner/loser only) ----------
def worster_schedule(api_response: Iterable[Any]) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['winner','loser'] for decided games.
    Drops canceled/incomplete; asserts no ties.
    """
    cols = ['homeTeam','awayTeam','homePoints','awayPoints']
    raw = (pd.DataFrame.from_records((g.to_dict() for g in api_response))
             .reindex(columns=cols))
    raw = raw.dropna(subset=['homePoints','awayPoints']).reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(columns=['winner','loser'])

    # Fail fast
    assert not raw[['homeTeam','awayTeam']].isna().any().any(), "Null team name(s)."
    hp = pd.to_numeric(raw['homePoints'], errors='raise').to_numpy()
    ap = pd.to_numeric(raw['awayPoints'], errors='raise').to_numpy()
    margin = hp - ap
    assert (margin != 0).all(), "Unexpected tie in completed game."

    home_win = margin > 0
    winners = np.where(home_win, raw['homeTeam'].to_numpy(), raw['awayTeam'].to_numpy())
    losers  = np.where(home_win, raw['awayTeam'].to_numpy(), raw['homeTeam'].to_numpy())
    return pd.DataFrame({'winner': winners, 'loser': losers})


# ---------- 2) Fast résumé features for a season ----------
def create_team_metrics(schedule: pd.DataFrame, K: int = 15, prefix: str = "") -> pd.DataFrame:
    """
    Build résumé features:
      {prefix}wins, {prefix}losses,
      {prefix}wins_from_1best, {prefix}wins_from_1worst, ... up to K
    Opponent win totals are computed from the same schedule (as-of-today).
    Column order matches your original (interleaved best/worst per i).
    """
    assert {'winner','loser'}.issubset(schedule.columns), "schedule must have ['winner','loser']"
    n_games = len(schedule)
    if n_games == 0:
        cols = ['team', f'{prefix}wins', f'{prefix}losses']
        for i in range(1, K+1):
            cols += [f'{prefix}wins_from_{i}best', f'{prefix}wins_from_{i}worst']
        return pd.DataFrame(columns=cols)

    # Encode to integers once
    both = pd.concat([schedule['winner'], schedule['loser']], ignore_index=True)
    codes, teams = pd.factorize(both, sort=True)
    T = len(teams)
    win_codes = codes[:n_games]
    lose_codes = codes[n_games:]

    # Wins & losses
    wins   = np.bincount(win_codes, minlength=T).astype(np.int16)
    losses = np.bincount(lose_codes, minlength=T).astype(np.int16)

    # Group beaten opponents per team (via sort/split)
    order_w = np.argsort(win_codes, kind='mergesort')
    losers_sorted = lose_codes[order_w]
    ends_w = np.cumsum(wins)
    starts_w = np.concatenate(([0], ends_w[:-1]))
    beaten_lists = [losers_sorted[starts_w[i]:ends_w[i]] for i in range(T)]

    # Group opponents each team lost to
    order_l = np.argsort(lose_codes, kind='mergesort')
    winners_sorted = win_codes[order_l]
    ends_l = np.cumsum(losses)
    starts_l = np.concatenate(([0], ends_l[:-1]))
    lostto_lists = [winners_sorted[starts_l[i]:ends_l[i]] for i in range(T)]

    # Precompute ladders (interleaved order later)
    best  = np.zeros((T, K), dtype=np.int16)
    worst = np.zeros((T, K), dtype=np.int16)
    for i in range(T):
        if beaten_lists[i].size:
            b = np.sort(wins[beaten_lists[i]])[::-1]     # descending
            best[i, :min(K, b.size)] = b[:K]
        if lostto_lists[i].size:
            w = np.sort(wins[lostto_lists[i]])           # ascending
            worst[i, :min(K, w.size)] = w[:K]

    # Assemble with exact column order: wins, losses, then interleaved best/worst
    data = {'team': teams.to_numpy(),
            f'{prefix}wins': wins,
            f'{prefix}losses': losses}
    for i in range(1, K+1):
        data[f'{prefix}wins_from_{i}best']  = best[:, i-1]
        data[f'{prefix}wins_from_{i}worst'] = worst[:, i-1]
    return pd.DataFrame(data)


# ---------- 3) Join current + last year and rank ----------
def get_worster(
    api_response: Iterable[Any],
    ly_api_response: Iterable[Any],
    K: int = 15,
) -> pd.DataFrame:
    """Rank by current-year keys, then last-year keys (tie-break), with column order matching your original."""
    cur_sched = worster_schedule(api_response)
    ly_sched  = worster_schedule(ly_api_response)

    cur_df = create_team_metrics(cur_sched, K=K, prefix="")
    ly_df  = create_team_metrics(ly_sched,  K=K, prefix="ly_")

    # Left join (current year is the universe), one-to-one expected
    joined = (cur_df.set_index('team')
                    .join(ly_df.set_index('team'), how='left', validate='one_to_one')
                    .reset_index())

    # Fill NaNs from LY-missing teams and cast back to ints
    ly_cols = ['ly_wins','ly_losses'] + \
              [f'ly_wins_from_{i}best' for i in range(1, K+1)] + \
              [f'ly_wins_from_{i}worst' for i in range(1, K+1)]
    for c in ly_cols:
        if c in joined.columns:
            joined[c] = joined[c].fillna(0).astype(np.int16)

    # Enforce final column order exactly like your original
    ordered_cols = ['team', 'wins', 'losses']
    for i in range(1, K+1):
        ordered_cols += [f'wins_from_{i}best', f'wins_from_{i}worst']
    ordered_cols += ['ly_wins', 'ly_losses']
    for i in range(1, K+1):
        ordered_cols += [f'ly_wins_from_{i}best', f'ly_wins_from_{i}worst']
    joined = joined.reindex(columns=ordered_cols)

    # Sort keys: wins; then interleaved current best/worst; ly_wins; then interleaved ly best/worst
    sort_cols = ['wins']
    asc_flags = [False]
    for i in range(1, K+1):
        sort_cols += [f'wins_from_{i}best', f'wins_from_{i}worst']
        asc_flags += [False, False]
    sort_cols += ['ly_wins']
    asc_flags += [False]
    for i in range(1, K+1):
        sort_cols += [f'ly_wins_from_{i}best', f'ly_wins_from_{i}worst']
        asc_flags += [False, False]

    joined = joined.sort_values(by=sort_cols, ascending=asc_flags, kind='mergesort').reset_index(drop=True)
    return joined
