import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from elo import load_players, compute_ratings_and_history, add_match
from statsmodels.nonparametric.smoothers_lowess import lowess


st.set_page_config(page_title="Pickleball Elo Ratings", layout="wide")
st.title("🏓 Pickleball Elo Tracker")

ratings, history, matches = compute_ratings_and_history()
players = sorted(load_players())

# Sidebar: Match Entry
st.sidebar.header("➕ Add Match Result")
col1, col2 = st.sidebar.columns(2)
p1 = col1.selectbox("Player 1", players)
p2 = col2.selectbox("Player 2", players)
col3, col4 = st.sidebar.columns(2)
score1 = col3.number_input(f"{p1} score", 0, 21, value=11)
score2 = col4.number_input(f"{p2} score", 0, 21, value=8)

# Password input
password = st.sidebar.text_input("🔒 Admin Password", type="password")

if st.sidebar.button("Submit Match"):
    if password == "letmein":  # change this to your actual secret
        msg = add_match(p1, p2, score1, score2)
        st.sidebar.success(msg)
        st.rerun()
    else:
        st.sidebar.error("Incorrect password! 🚫")

# Ratings Table
st.header("📊 Current Elo Ratings")
df = pd.DataFrame(ratings.items(), columns=["Player", "Rating"])
df = df.sort_values(by="Rating", ascending=False).reset_index(drop=True)
st.dataframe(df.style.format({"Rating": "{:.2f}"}), use_container_width=True)

# Graph
# Convert history to long-form DataFrame for seaborn
# Plot using seaborn

# Graph
# Graph
st.header("📈 Elo Progress Over Matches")

# Convert history to long-form DataFrame for seaborn
graph_data = []
for player, series in history.items():
    for match_num, rating in series:
        graph_data.append({"Player": player, "Match #": match_num, "Elo Rating": rating})

graph_df = pd.DataFrame(graph_data)

# Plot
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(14, 6))

sns.lineplot(
    data=graph_df,
    x="Match #",
    y="Elo Rating",
    hue="Player",
    marker="o",
    linewidth=2.5,
    markersize=8,
    ax=ax
)

# Make it pretty
ax.set_title("🔥 Elo Ratings Per Match", fontsize=18, weight="bold", pad=15)
ax.set_xlabel("Match #", fontsize=12)
ax.set_ylabel("Elo Rating", fontsize=12)
ax.legend(title="Player", bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
sns.despine(left=False, bottom=False)

# Optional: lighten grid, adjust spacing
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

st.pyplot(fig)



# 🎯 Power Rankings & Elo Tiers
st.header("🏅 Power Rankings & Tiers")

# --- Streaks Dictionary ---
# From earlier section where you compute win/loss streaks
# streaks[player] = +ve (win streak), -ve (loss streak), 0 (no streak)
# You must have `streaks` already calculated, if not:
from collections import defaultdict

def get_current_streaks(matches):
    streaks = {}
    
    # Store player-wise result history (latest first)
    player_results = defaultdict(list)

    for match in reversed(matches):  # latest first
        results = [
            (match["player1"], match["score1"], match["score2"]),
            (match["player2"], match["score2"], match["score1"]),
        ]
        for player, score, opp_score in results:
            result = "W" if score > opp_score else "L"
            player_results[player].append(result)

    for player, results in player_results.items():
        if not results:
            streaks[player] = 0
            continue

        streak_type = results[0]  # W or L
        streak_count = 0
        for result in results:
            if result == streak_type:
                streak_count += 1
            else:
                break

        # Make it negative if it’s a losing streak
        streaks[player] = streak_count if streak_type == "W" else -streak_count

    return streaks


streaks = get_current_streaks(matches)

# --- Power Ranking ---
def compute_power_ranking(rating, streak):
    return rating + (5 * streak)

# --- Elo Tiers ---
def elo_tier_classification(elo):
    if elo >= 1050:
        return "🏆 Elite"
    elif elo >= 1020:
        return "🔥 Pro"
    elif elo >= 990:
        return "📈 Rising Star"
    else:
        return "🧱 Developing"

# Combine into DataFrame
df_power = pd.DataFrame([
    {
        "Player": player,
        "Elo Rating": ratings[player],
        "Current Streak": f"{abs(streak)}{'W' if streak > 0 else 'L' if streak < 0 else ''}",
        "Power Ranking": compute_power_ranking(ratings[player], streak),
        "Tier": elo_tier_classification(ratings[player])
    }
    for player, streak in streaks.items()
])

df_power = df_power.sort_values(by="Power Ranking", ascending=False).reset_index(drop=True)

st.dataframe(df_power.style.format({
    "Elo Rating": "{:.2f}",
    "Power Ranking": "{:.2f}"
}), use_container_width=True)




# Match history
st.header("📜 Match History")
match_df = pd.DataFrame(matches)
if not match_df.empty:
    st.dataframe(match_df[::-1], use_container_width=True)
else:
    st.write("No matches yet.")

from collections import defaultdict

st.header("📊 Player Performance Stats")

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
stats_df = stats_df.sort_values(by="Wins", ascending=False).reset_index(drop=True)
st.dataframe(stats_df, use_container_width=True)


st.header("🤜🤛 Head-to-Head Record")

# Get list of players
player_set = set()
for match in matches:
    player_set.update([match["player1"], match["player2"]])
players = sorted(list(player_set))

# Initialize H2H win matrix
h2h = pd.DataFrame(0, index=players, columns=players)

# Populate matrix
for match in matches:
    p1, p2 = match["player1"], match["player2"]
    s1, s2 = match["score1"], match["score2"]

    if s1 > s2:
        winner, loser = p1, p2
    else:
        winner, loser = p2, p1

    h2h.loc[winner, loser] += 1

# Styling the table
def highlight_diagonal(val):
    return 'background-color: lightgray' if val.name == val.index else ''

st.dataframe(h2h.style.format("{:.0f}").set_caption("Wins Against Other Players").background_gradient(cmap="Blues", axis=None), use_container_width=True)

