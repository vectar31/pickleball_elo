import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from elo import load_players, compute_ratings_and_history, add_match
from statsmodels.nonparametric.smoothers_lowess import lowess
from elo import compute_doubles_ratings_and_history
from statsmodels.nonparametric.smoothers_lowess import lowess



def sort_with_promoter_last(df, sort_by, ascending=False):
    df_sorted = df.sort_values(by=sort_by, ascending=ascending)
    if "Piyush" in df_sorted["Player"].values:
        piyush_row = df_sorted[df_sorted["Player"] == "Piyush"]
        df_sorted = df_sorted[df_sorted["Player"] != "Piyush"]
        df_sorted = pd.concat([df_sorted, piyush_row], ignore_index=True)
    return df_sorted

st.set_page_config(page_title="Pickleball Elo Ratings", layout="wide")
st.title("🏓 Pickleball Elo Tracker")

ratings, history, matches = compute_ratings_and_history()
doubles_ratings, doubles_history, doubles_matches = compute_doubles_ratings_and_history()

# Set of players who have played at least one match
active_players = set()
for match in matches:
    active_players.update([match["player1"], match["player2"]])

players = sorted(load_players())

# Sidebar: Match Entry
# Sidebar: Club Stats Overview
st.sidebar.header("📊 Club Stats")

# Basic counts
total_singles_matches = len(matches)
total_doubles_matches = len(doubles_matches)

singles_players = set()
for match in matches:
    singles_players.update([match["player1"], match["player2"]])

doubles_players = set()
for match in doubles_matches:
    doubles_players.update(match["team1"] + match["team2"])

# Display stats
st.sidebar.markdown(f"""
- 🧑 **Total Club Members:** {len(set(load_players()))}
- 🏓 **Singles Players:** {len(singles_players)}
- 👯 **Doubles Players:** {len(doubles_players)}
- 🎮 **Singles Matches Played:** {total_singles_matches}
- 🎮 **Doubles Matches Played:** {total_doubles_matches}
- 🔁 **Total Matches:** {total_singles_matches + total_doubles_matches}
""")


# Ratings Table
st.header("📊 Singles Elo Ratings")
df = pd.DataFrame([
    (p, ratings[p]) for p in ratings if p in active_players
], columns=["Player", "Rating"])
# df = sort_with_promoter_last(df, sort_by="Rating", ascending=False)
df = df.sort_values(by="Rating", ascending=False).reset_index(drop=True)
st.dataframe(df.style.format({"Rating": "{:.2f}"}), use_container_width=True)

# Graph


# Convert history to long-form DataFrame for seaborn
# Determine max match #
max_match_num = max([match_num for series in history.values() for match_num, _ in series])

# Pad history with last Elo
graph_data = []
for player, series in history.items():
    if player not in active_players or not series:
        continue

    # Add actual history
    # Interpolate between matches
    for i in range(len(series)):
        match_num, rating = series[i]
        graph_data.append({"Player": player, "Match #": match_num, "Elo Rating": rating})

        # Add flat Elo for skipped matches (if any)
        if i < len(series) - 1:
            next_match_num, _ = series[i + 1]
            for skipped in range(match_num + 1, next_match_num):
                graph_data.append({
                    "Player": player,
                    "Match #": skipped,
                    "Elo Rating": rating  # same as current
                })


    # Add extension to max match
    last_match_num, last_rating = series[-1]
    if last_match_num < max_match_num:
        graph_data.append({
            "Player": player,
            "Match #": max_match_num,
            "Elo Rating": last_rating
        })



graph_df = pd.DataFrame(graph_data)

# Optional: Detailed Elo graph for a single player
st.subheader("🔍 Singles Elo History (One Player at a Time)")

# Get unique players
unique_singles_players = sorted(graph_df["Player"].unique())
selected_single_player = st.selectbox("Select a player to view their Elo trend:", unique_singles_players, key="singles_player_smooth")

# Filter and sort player's data
single_player_df = graph_df[graph_df["Player"] == selected_single_player].sort_values("Match #")

# Create a new match number sequence just for this player's matches
player_matches = single_player_df[single_player_df["Elo Rating"] != single_player_df["Elo Rating"].shift()]  # Get only actual matches
player_matches = player_matches.reset_index(drop=True)
player_matches["Player Match #"] = player_matches.index + 1

# Merge the new match numbers back to the full sequence
single_player_df = single_player_df.merge(
    player_matches[["Match #", "Player Match #"]], 
    on="Match #", 
    how="left"
)
single_player_df["Player Match #"] = single_player_df["Player Match #"].fillna(method='ffill')

# Apply LOESS smoothing to player's Elo trend
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    single_player_df["Player Match #"],
    single_player_df["Elo Rating"],
    marker="o",
    linewidth=2,
    color="#67cfff",
    label="Elo Rating"
)

# # Fill under the curve
# ax.fill_between(
#     smoothed[:, 0],
#     smoothed[:, 1],
#     alpha=0.2,
#     color="#67cfff"
# )

# Style
ax.set_title(f"📈 Elo Progress: {selected_single_player}", fontsize=16, weight="bold")
ax.set_xlabel("Player's Match #", fontsize=12)  # Changed label to reflect player-specific matches
ax.set_ylabel("Elo Rating", fontsize=12)

# Dynamic Y-axis limits
# elo_min = smoothed[:, 1].min()
# elo_max = smoothed[:, 1].max()
# buffer = max(10, (elo_max - elo_min) * 0.1)
# ax.set_ylim(elo_min - buffer, elo_max + buffer)

# Dark styling
ax.grid(alpha=0.3)
ax.set_facecolor("#1e1e1e")
fig.patch.set_facecolor("#1e1e1e")
ax.tick_params(colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')

st.pyplot(fig)


st.markdown("""
<hr style="border: none; height: 2px; background-color: #FF4B4B; margin: 20px 0;">
""", unsafe_allow_html=True)

# 🔁 Doubles Elo Ratings
st.header("👯 Doubles Elo Ratings")

doubles_data = [
    {"Player": p, "Doubles Elo": r}
    for p, r in doubles_ratings.items()
    if len(doubles_history[p]) > 1
]

df_doubles = pd.DataFrame(doubles_data)
df_doubles = df_doubles.sort_values(by="Doubles Elo", ascending=False).reset_index(drop=True)

if df_doubles.empty:
    st.write("No doubles matches played yet.")
else:
    st.dataframe(df_doubles.style.format({"Doubles Elo": "{:.2f}"}), use_container_width=True)



# Determine max match number for doubles
max_doubles_match_num = max(
    [match_num for series in doubles_history.values() for match_num, _ in series]
    or [0]  # fallback if empty
)

# Build extended and interpolated doubles Elo history
doubles_graph_data = []
for player, series in doubles_history.items():
    if not series:
        continue

    # Add match-by-match Elo, interpolating missed matches
    for i in range(len(series)):
        match_num, rating = series[i]
        doubles_graph_data.append({
            "Player": player,
            "Match #": match_num,
            "Doubles Elo": rating
        })

        # Interpolate Elo for skipped matches (flat line)
        if i < len(series) - 1:
            next_match_num, _ = series[i + 1]
            for skipped_match in range(match_num + 1, next_match_num):
                doubles_graph_data.append({
                    "Player": player,
                    "Match #": skipped_match,
                    "Doubles Elo": rating
                })

    # Extend to latest match if needed
    last_match_num, last_rating = series[-1]
    for extra_match in range(last_match_num + 1, max_doubles_match_num + 1):
        doubles_graph_data.append({
            "Player": player,
            "Match #": extra_match,
            "Doubles Elo": last_rating
        })


doubles_graph_df = pd.DataFrame(doubles_graph_data)

# Optional: Detailed Elo graph for a single player
from statsmodels.nonparametric.smoothers_lowess import lowess

st.subheader("🔍 Doubles Elo History (One Player at a Time)")

# Get unique players
unique_doubles_players = sorted(doubles_graph_df["Player"].unique())
selected_player = st.selectbox("Select a player to view their Elo trend:", unique_doubles_players, key="doubles_player_smooth")

# Filter and sort player's data
player_df = doubles_graph_df[doubles_graph_df["Player"] == selected_player].sort_values("Match #")

# Create a new match number sequence just for this player's matches
player_matches = player_df[player_df["Doubles Elo"] != player_df["Doubles Elo"].shift()]  # Get only actual matches
player_matches = player_matches.reset_index(drop=True)
player_matches["Player Match #"] = player_matches.index + 1

# Merge the new match numbers back to the full sequence
player_df = player_df.merge(
    player_matches[["Match #", "Player Match #"]], 
    on="Match #", 
    how="left"
)
player_df["Player Match #"] = player_df["Player Match #"].fillna(method='ffill')

# Apply LOESS smoothing
smoothed = lowess(
    exog=player_df["Player Match #"],
    endog=player_df["Doubles Elo"],
    frac=0.3  # adjust for more/less smoothing
)

# Plot smoothed Elo
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    smoothed[:, 0],
    smoothed[:, 1],
    linewidth=2.5,
    color="#67cfff",
    label="Smoothed Elo Rating"
)

# Fill under the curve
ax.fill_between(
    smoothed[:, 0],
    smoothed[:, 1],
    alpha=0.2,
    color="#67cfff"
)

# Style tweaks
ax.set_title(f"📈 Elo Progress: {selected_player}", fontsize=16, weight="bold")
ax.set_xlabel("Player's Match #", fontsize=12)  # Changed label to reflect player-specific matches
ax.set_ylabel("Doubles Elo", fontsize=12)

# Set x-axis to show only integer ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Dynamic Y-axis limits
elo_min = smoothed[:, 1].min()
elo_max = smoothed[:, 1].max()
buffer = max(10, (elo_max - elo_min) * 0.1)
ax.set_ylim(elo_min - buffer, elo_max + buffer)

# Dark mode style
ax.grid(alpha=0.3)
ax.set_facecolor("#1e1e1e")
fig.patch.set_facecolor("#1e1e1e")
ax.tick_params(colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')

st.pyplot(fig)



st.markdown("""
<hr style="border: none; height: 2px; background-color: #FF4B4B; margin: 20px 0;">
""", unsafe_allow_html=True)

from collections import defaultdict

st.header("📊Singles Player Performance Stats")

# Initialize stat containers
stats = defaultdict(lambda: {
    "Wins": 0,
    "Losses": 0,
    "Games": 0,
    "Points Won": 0,
    "Points Lost": 0,
    "Current Streak": [],
    "Streak History": []
})

# Populate stats
for match in matches:
    p1, p2 = match["player1"], match["player2"]
    s1, s2 = match["score1"], match["score2"]

    # Determine winner and loser
    if s1 > s2:
        winner, loser = p1, p2
        w_score, l_score = s1, s2
    else:
        winner, loser = p2, p1
        w_score, l_score = s2, s1

    # Update wins/losses
    stats[winner]["Wins"] += 1
    stats[loser]["Losses"] += 1

    # Update points
    stats[p1]["Points Won"] += s1
    stats[p1]["Points Lost"] += s2
    stats[p2]["Points Won"] += s2
    stats[p2]["Points Lost"] += s1

    # Update games
    stats[p1]["Games"] += 1
    stats[p2]["Games"] += 1

    # Update streaks
    for player in [p1, p2]:
        result = "W" if player == winner else "L"
        stats[player]["Streak History"].append(result)

# Calculate additional stats
processed_stats = []
for player, data in stats.items():
    if player not in active_players:
        continue
    total_mathches = data["Wins"] + data["Losses"]
    wins = data["Wins"]
    losses = data["Losses"]
    games = data["Games"]
    pw = data["Points Won"]
    pl = data["Points Lost"]
    history = data["Streak History"]

    # Current streak
    current_streak = ""
    if history:
        last = history[-1]
        count = 0
        for res in reversed(history):
            if res == last:
                count += 1
            else:
                break
        current_streak = f"{count}{last}"

    # Longest win/loss streak
    def max_streak(seq, target):
        max_count = count = 0
        for res in seq:
            if res == target:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    longest_w = max_streak(history, "W")
    longest_l = max_streak(history, "L")

    processed_stats.append({
        "Player": player,
        "Matches": total_mathches,
        "Wins": wins,
        "Losses": losses,
        "W/L %": round(wins / games * 100, 1) if games > 0 else 0,
        "Current Streak": current_streak,
        "Longest Win Streak": longest_w,
        "Longest Loss Streak": longest_l,
        "Avg Points Won": round(pw / games, 1) if games > 0 else 0,
        "Avg Points Lost": round(pl / games, 1) if games > 0 else 0,
    })

# Display as DataFrame
stats_df = pd.DataFrame(processed_stats)
stats_df = stats_df.sort_values("Wins", ascending=False).reset_index(drop=True)
st.dataframe(stats_df, use_container_width=True)

st.header("📊Doubles Player Performance Stats")

# Initialize doubles stat containers
doubles_stats = defaultdict(lambda: {
    "Wins": 0,
    "Losses": 0,
    "Games": 0,
    "Points Won": 0,
    "Points Lost": 0,
    "Streak History": []
})

# Populate doubles stats
for match in doubles_matches:
    team1 = match["team1"]
    team2 = match["team2"]
    s1, s2 = match["score1"], match["score2"]

    # Determine winner and loser
    if s1 > s2:
        winners = team1
        losers = team2
        w_score, l_score = s1, s2
    else:
        winners = team2
        losers = team1
        w_score, l_score = s2, s1

    for player in winners:
        doubles_stats[player]["Wins"] += 1
        doubles_stats[player]["Streak History"].append("W")

    for player in losers:
        doubles_stats[player]["Losses"] += 1
        doubles_stats[player]["Streak History"].append("L")

    for player in team1 + team2:
        doubles_stats[player]["Games"] += 1

    for player in team1:
        doubles_stats[player]["Points Won"] += s1
        doubles_stats[player]["Points Lost"] += s2

    for player in team2:
        doubles_stats[player]["Points Won"] += s2
        doubles_stats[player]["Points Lost"] += s1

# Process stats
processed_doubles_stats = []
for player, data in doubles_stats.items():
    if player not in doubles_ratings:
        continue
    total_matches = data["Wins"] + data["Losses"]
    games = data["Games"]
    wins = data["Wins"]
    losses = data["Losses"]
    pw = data["Points Won"]
    pl = data["Points Lost"]
    history = data["Streak History"]

    # Current streak
    current_streak = ""
    if history:
        last = history[-1]
        count = 0
        for res in reversed(history):
            if res == last:
                count += 1
            else:
                break
        current_streak = f"{count}{last}"

    # Longest win/loss streak
    def max_streak(seq, target):
        max_count = count = 0
        for res in seq:
            if res == target:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    longest_w = max_streak(history, "W")
    longest_l = max_streak(history, "L")

    processed_doubles_stats.append({
        "Player": player,
        "Matches": total_matches,
        "Wins": wins,
        "Losses": losses,
        "W/L %": round(wins / games * 100, 1) if games > 0 else 0,
        "Current Streak": current_streak,
        "Longest Win Streak": longest_w,
        "Longest Loss Streak": longest_l,
        "Avg Points Won": round(pw / games, 1) if games > 0 else 0,
        "Avg Points Lost": round(pl / games, 1) if games > 0 else 0,
    })

# Display
doubles_stats_df = pd.DataFrame(processed_doubles_stats)
doubles_stats_df = doubles_stats_df.sort_values("Wins", ascending=False).reset_index(drop=True)
st.dataframe(doubles_stats_df, use_container_width=True)


# 🤝 Doubles Partnership Insights
st.header("🤝 Doubles Partner Synergy & Matchup Stats")

# Track partner stats
partner_stats = defaultdict(lambda: defaultdict(lambda: {"matches": 0, "wins": 0}))
matchup_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0}))

# Process match data
for match in doubles_matches:
    t1, t2 = match["team1"], match["team2"]
    s1, s2 = match["score1"], match["score2"]

    if s1 > s2:
        winning_team = t1
        losing_team = t2
    else:
        winning_team = t2
        losing_team = t1

    # Partner stats
    for team in [t1, t2]:
        for p1 in team:
            for p2 in team:
                if p1 != p2:
                    partner_stats[p1][p2]["matches"] += 1
                    if team == winning_team:
                        partner_stats[p1][p2]["wins"] += 1

    # Matchup stats
    team1_key = " & ".join(sorted(t1))
    team2_key = " & ".join(sorted(t2))
    if s1 > s2:
        matchup_stats[team1_key][team2_key]["wins"] += 1
        matchup_stats[team1_key][team2_key]["total"] += 1
        matchup_stats[team2_key][team1_key]["losses"] += 1
        matchup_stats[team2_key][team1_key]["total"] += 1
    else:
        matchup_stats[team2_key][team1_key]["wins"] += 1
        matchup_stats[team2_key][team1_key]["total"] += 1
        matchup_stats[team1_key][team2_key]["losses"] += 1
        matchup_stats[team1_key][team2_key]["total"] += 1

# Best Partner Table
best_partner_data = []
for player, partners in partner_stats.items():
    best = max(partners.items(), key=lambda x: x[1]["wins"], default=(None, {"wins": 0}))
    best_partner, stats = best
    total = stats["matches"]
    wins = stats["wins"]
    win_pct = round(100 * wins / total, 1) if total > 0 else 0
    best_partner_data.append({
        "Player": player,
        "Best Partner": best_partner,
        "Matches Together": total,
        "Wins Together": wins,
        "Win %": win_pct
    })

best_partner_df = pd.DataFrame(best_partner_data).sort_values(by="Win %", ascending=False)
st.subheader("🏅 Best Doubles Partner (by Win %)")
st.dataframe(best_partner_df, use_container_width=True)

# Matchup Stats Table
matchup_data = []
for team1, opponents in matchup_stats.items():
    for team2, stats in opponents.items():
        if team1 < team2:  # avoid duplicate reverse entries
            matchup_data.append({
                "Team 1": team1,
                "Team 2": team2,
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Total Matches": stats["total"],
                "Win %": round(100 * stats["wins"] / stats["total"], 1) if stats["total"] > 0 else 0
            })

matchup_df = pd.DataFrame(matchup_data).sort_values(by="Total Matches", ascending=False)

st.subheader("🏓 Doubles Matchup Records")
st.dataframe(matchup_df, use_container_width=True)




st.header("🤜🤛 Singles Head-to-Head Record")

# === Build H2H Matrix ===
players = sorted(list(active_players))
h2h_wins = pd.DataFrame(0, index=players, columns=players)
h2h_losses = pd.DataFrame(0, index=players, columns=players)

# Count wins/losses
for match in matches:
    p1, p2 = match["player1"], match["player2"]
    s1, s2 = match["score1"], match["score2"]
    if p1 not in players or p2 not in players:
        continue
    if s1 > s2:
        winner, loser = p1, p2
    else:
        winner, loser = p2, p1
    h2h_wins.loc[winner, loser] += 1
    h2h_losses.loc[loser, winner] += 1

# === PART 1: Interactive View (Player vs All) ===
st.subheader("🔍 Select a Player to View Matchups")
selected = st.selectbox("Choose a player:", players, key="h2h_selector")

one_vs_all = []
for opponent in players:
    if opponent == selected:
        continue
    wins = h2h_wins.loc[selected, opponent]
    losses = h2h_losses.loc[selected, opponent]
    total = wins + losses
    win_pct = round(100 * wins / total, 1) if total > 0 else np.nan
    one_vs_all.append({
        "Opponent": opponent,
        "Wins": wins,
        "Losses": losses,
        "Total Matches": total,
        "Win %": win_pct
    })

one_vs_all_df = pd.DataFrame(one_vs_all)

# Drop rows with 0 matches
one_vs_all_df = one_vs_all_df[one_vs_all_df["Total Matches"] > 0]

# Sort by win %
one_vs_all_df = one_vs_all_df.sort_values(by="Win %", ascending=False, na_position="last")

st.dataframe(one_vs_all_df, use_container_width=True)

# Singles Match history
with st.expander("📜 Singles Match History", expanded=False):
    st.markdown("## 📜 Singles Match History")
    match_df = pd.DataFrame(matches)
    if not match_df.empty:
        st.dataframe(match_df[::-1], use_container_width=True)
    else:
        st.write("No matches yet.")

# 📜 Doubles Match History
with st.expander("📜 Doubles Match History", expanded=False):
    st.markdown("## 📜 Doubles Match History")
    if not doubles_matches:
        st.write("No doubles matches yet.")
    else:
        doubles_match_df = pd.DataFrame([
            {
                "Date": match["date"],
                "Team 1": " + ".join(match["team1"]),
                "Score 1": match["score1"],
                "Score 2": match["score2"],
                "Team 2": " + ".join(match["team2"]),
            }
            for match in doubles_matches
        ])
        st.dataframe(doubles_match_df[::-1], use_container_width=True)


with st.expander("🧑‍🤝‍🧑 Club Members", expanded=False):
    st.markdown("## 🧑‍🤝‍🧑 Club Members")
    # All registered players
    all_players = sorted(load_players())  # from input file
    rated_players = active_players       # already computed
    unrated_players = set(all_players) - rated_players
    
    # Build display DataFrame
    members_df = pd.DataFrame([
        {
            "Player": player,
            "Status": "🟢 Rated" if player in rated_players else "⚪ Unrated"
        }
        for player in all_players
    ])
    
    # Show table
    st.dataframe(members_df, use_container_width=True)
