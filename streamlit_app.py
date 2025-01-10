import streamlit as st
import pymc as pm
import pandas as pd
import requests
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Hockey Simulation", layout="wide")

# Constants
SAMPLES = 1000
TUNE = 1000

# Helper functions
@st.cache_data
def get_data(url):
    response = requests.get(url).json()
    result = pd.DataFrame.from_dict(response['data'])
    while response.get('links', {}).get('next'):
        response = requests.get(response['links']['next'] + '&show_all=1').json()
        result = pd.concat([result, pd.DataFrame.from_dict(response['data'])])
    return result

def fetch_and_process_data(team, competition, subcompetition):
    if subcompetition != "":
        subcomp_string = f"&competition_id={subcompetition}"
    else:
        subcomp_string = ""
    upcoming_games = get_data(f"https://publicaties.hockeyweerelt.nl/mc/teams/{competition}/matches/upcoming?&show_all=1{subcomp_string}")
    played_games = get_data(f"https://publicaties.hockeyweerelt.nl/mc/teams/{competition}/matches/official?=&show_all=1{subcomp_string}")
    standing_raw = get_data(f"https://publicaties.hockeyweerelt.nl/mc/teams/{competition}/standing")

    comp = pd.concat([upcoming_games, played_games])
    comp['hometeam'] = comp['home_team'].str['name']
    comp['awayteam'] = comp['away_team'].str['name']
    comp = comp[['datetime', 'hometeam', 'awayteam', 'home_score', 'away_score']]

    standing = pd.DataFrame.from_dict(standing_raw['standings'][0])
    standing['team'] = standing['team'].str['name']

    category_order = standing['team']
    comp.hometeam = comp.hometeam.astype("category").cat.set_categories(category_order)
    comp.awayteam = comp.awayteam.astype("category").cat.set_categories(category_order)
    comp[['played']] = ~pd.isnull(comp[['home_score']])

    return comp, standing

# Streamlit App Layout
st.title("Hockey Simulation")

# User Inputs
team = st.text_input("Enter Team Name:", "Were Di H5")
competition = st.text_input("Enter Competition ID:", "N7816")
subcompetition = st.text_input("Enter Subcompetition ID (Optional):", "")

if st.button("Run Simulation"):
    with st.spinner("Fetching data and running simulation..."):
        comp, standing = fetch_and_process_data(team, competition, subcompetition)

        comp['played'] = ~pd.isnull(comp[['home_score']])
        played = comp[comp['played']]
        not_played = comp[~comp['played']]

        n_teams = len(standing['team'])
        scores = pd.concat([played['home_score'], played['away_score']], ignore_index=True)

        with pm.Model() as model:
            hometeam = pm.Data("hometeam", played.hometeam.cat.codes.astype('int8').values)
            awayteam = pm.Data("awayteam", played.awayteam.cat.codes.astype('int8').values)
            hyper = pm.Normal('hyper', np.log(np.mean(scores)), 0.5)

            attack_rate = pm.Normal('attack_rate', 0, 1, shape=n_teams)
            defense_rate = pm.Normal('defense_rate', 0, 1, shape=n_teams)

            home_diff = (attack_rate[hometeam] - defense_rate[awayteam]) + hyper
            away_diff = (attack_rate[awayteam] - defense_rate[hometeam]) + hyper
            home_goals = pm.Poisson('home_goals', pm.math.exp(home_diff), observed=played.home_score.values.astype(int))
            away_goals = pm.Poisson('away_goals', pm.math.exp(away_diff), observed=played.away_score.values.astype(int))

            trace = pm.sample(SAMPLES, tune=TUNE)

        # Display Results
        st.subheader("Simulation Results")
        figs = pm.plot_trace(trace)

        st.pyplot(figs.flatten()[0].figure)

        team_names = comp.hometeam.cat.categories
        attack_rate = trace.posterior.attack_rate[1].to_pandas().median()
        defense_rate = trace.posterior.defense_rate[1].to_pandas().median()

        df = pd.concat([attack_rate, defense_rate], axis=1)
        df.columns = ['attack_rate', 'defense_rate']
        df.index = team_names
        df.sort_values("attack_rate", inplace=True)

        for column in df.columns:
            fig = px.bar(df, x=df.index, y=column, title=f'Bar Plot for {column}')
            st.plotly_chart(fig, use_container_width=True)

        with model:
            ppc = pm.sample_posterior_predictive(trace)

    MAE = (np.mean(abs(played['home_score'] - np.median(ppc.posterior_predictive['home_goals'][1], axis=0))) +
           np.mean(abs(played['away_score'] - np.median(ppc.posterior_predictive['away_goals'][1], axis=0)))) / 2

    st.success(f'MAE: {MAE:.4f}')

    upcoming_games = comp[~comp.played].reset_index()
    # Determine the number of upcoming games
    n_upcoming_games = len(upcoming_games)

    # Create arrays for hometeam and awayteam for upcoming games
    upcoming_hometeam = upcoming_games.hometeam.cat.codes.astype('int8').values
    upcoming_awayteam = upcoming_games.awayteam.cat.codes.astype('int8').values

    # Create arrays for hyper from the trace
    hyper_samples = trace.posterior.hyper[1]

    # Predict goals for upcoming games
    upcoming_home_diff = (
            (trace.posterior.attack_rate[1][:, upcoming_hometeam].T.values - trace.posterior.defense_rate[1][:,
                                                                             upcoming_awayteam].T.values) + hyper_samples.values
    )
    upcoming_away_diff = (
            (trace.posterior.attack_rate[1][:, upcoming_awayteam].T.values - trace.posterior.defense_rate[1][:,
                                                                             upcoming_hometeam].T.values) + hyper_samples.values
    )

    # Compute the Poisson parameter for upcoming games
    predictions = {
        "home_goals": np.random.poisson(np.exp(upcoming_home_diff)),
        "away_goals": np.random.poisson(np.exp(upcoming_away_diff))
    }

    goals_dict = {}
    for column in ['home_goals', 'away_goals']:
        goals = pd.DataFrame(predictions[column])
        goals[['hometeam']] = upcoming_games[['hometeam']]
        goals[['awayteam']] = upcoming_games[['awayteam']]
        goals[['datetime']] = upcoming_games[['datetime']]
        goals_dict[column] = pd.melt(goals, id_vars=['hometeam', 'awayteam', 'datetime'])
        goals_dict[column].columns = ['hometeam', 'awayteam', 'datetime', 'simulation', column]

    scores_df = goals_dict['home_goals'].merge(goals_dict['away_goals'],
                                               on=['hometeam', 'awayteam', 'simulation', 'datetime'])
    scores_df['home_win'] = scores_df['home_goals'] > scores_df['away_goals']
    scores_df['draw'] = scores_df['home_goals'] == scores_df['away_goals']
    scores_df['away_win'] = scores_df['home_goals'] < scores_df['away_goals']
    scores_df['home_points'] = scores_df['home_win'] * 3 + scores_df['draw']
    scores_df['away_points'] = scores_df['away_win'] * 3 + scores_df['draw']

    home_results = scores_df[['hometeam', 'simulation', 'home_points', 'home_goals', 'away_goals']].groupby(
        ['hometeam', 'simulation']).sum().reset_index()
    home_results.rename(columns={"home_goals": "goals_for_home", "away_goals": "goals_against_home"}, inplace=True)
    away_results = scores_df[['awayteam', 'simulation', 'away_points', 'home_goals', 'away_goals']].groupby(
        ['awayteam', 'simulation']).sum().reset_index()
    away_results.rename(columns={"home_goals": "goals_against_away", "away_goals": "goals_for_away"}, inplace=True)

    added_points = home_results.merge(
        away_results,
        left_on=['hometeam', 'simulation'],
        right_on=['awayteam', 'simulation']
    )
    added_points['added_points'] = added_points['home_points'] + added_points['away_points']
    added_points['added_goals_for'] = added_points['goals_for_home'] + added_points['goals_for_away']
    added_points['added_goals_against'] = added_points['goals_against_home'] + added_points['goals_against_away']
    added_points['team'] = added_points['hometeam']
    result = standing[['team', 'points', 'goals_for', 'goals_against']].merge(
        added_points[['team', 'simulation', 'added_points', 'added_goals_for', 'added_goals_against']])
    result['total'] = result['points'] + result['added_points']
    result['goals_for'] = result['goals_for'] + result['added_goals_for']
    result['goals_against'] = result['goals_against'] + result['added_goals_against']

    result = pd.concat([group.sort_values(by=['total', 'goals_for', 'goals_against'],
                                          ascending=[False, False, True]).assign(position=range(1, len(group) + 1))
                        for name, group in result.groupby('simulation')], ignore_index=True)

    # Calculate bar_input
    bar_input = (result[result['team'] == team]['position'].value_counts() / SAMPLES).sort_index()
