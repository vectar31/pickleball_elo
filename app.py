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

# Save original history for visualization (prevent overwriting in loops)
elo_history = history

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
max_match_num = max([match_num for series in elo_history.values() for match_num, _ in series])

# Pad history with last Elo
graph_data = []
for player, series in elo_history.items():
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


from collections import defaultdict

st.header("📊 Singles Player Performance Stats")

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

# Singles Match history
with st.expander("📜 Singles Match History", expanded=False):
    st.markdown("## 📜 Singles Match History")
    match_df = pd.DataFrame(matches)
    if not match_df.empty:
        st.dataframe(match_df[::-1], use_container_width=True)
    else:
        st.write("No matches yet.")


# ============================================================
# ADVANCED SINGLES VISUALIZATIONS
# ============================================================

st.markdown("""
<hr style="border: none; height: 2px; background-color: #4CAF50; margin: 30px 0;">
""", unsafe_allow_html=True)

st.header("📈 Advanced Singles Analytics")

# 6. PLAYER COMPARISON TOOL (Moved to top for quick access)
st.subheader("🔬 Player Comparison Tool")

col1, col2 = st.columns(2)

# Set default players (check if they exist in active players)
sorted_players = sorted(active_players)
default_player1 = "Arora" if "Arora" in active_players else sorted_players[0]
default_player2 = "Ameta" if "Ameta" in active_players else (sorted_players[1] if len(sorted_players) > 1 else sorted_players[0])

with col1:
    player1_compare = st.selectbox("Select Player 1:", sorted_players, 
                                   index=sorted_players.index(default_player1), 
                                   key="compare1")
with col2:
    player2_compare = st.selectbox("Select Player 2:", sorted_players, 
                                   index=sorted_players.index(default_player2), 
                                   key="compare2")

if player1_compare != player2_compare:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Get stats for both players
    p1_stats = next((s for s in processed_stats if s["Player"] == player1_compare), None)
    p2_stats = next((s for s in processed_stats if s["Player"] == player2_compare), None)
    
    if p1_stats and p2_stats:
        with col1:
            st.markdown(f"### {player1_compare}")
            st.metric("ELO", f"{ratings[player1_compare]:.0f}")
            st.metric("Wins", p1_stats["Wins"])
            st.metric("Win %", f"{p1_stats['W/L %']:.1f}%")
            st.metric("Avg Pts Won", p1_stats["Avg Points Won"])
        
        with col2:
            # Radar chart
            categories = ['ELO\n(normalized)', 'Win Rate', 'Avg Pts Won', 'Longest\nWin Streak', 'Matches\nPlayed']
            
            # Normalize values to 0-100 scale
            max_elo = max(ratings.values())
            min_elo = min(ratings.values())
            p1_values = [
                ((ratings[player1_compare] - min_elo) / (max_elo - min_elo)) * 100,
                p1_stats['W/L %'],
                (p1_stats['Avg Points Won'] / 11) * 100,
                min(p1_stats['Longest Win Streak'] / 10 * 100, 100),
                min(p1_stats['Matches'] / 50 * 100, 100)
            ]
            p2_values = [
                ((ratings[player2_compare] - min_elo) / (max_elo - min_elo)) * 100,
                p2_stats['W/L %'],
                (p2_stats['Avg Points Won'] / 11) * 100,
                min(p2_stats['Longest Win Streak'] / 10 * 100, 100),
                min(p2_stats['Matches'] / 50 * 100, 100)
            ]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            p1_values += p1_values[:1]
            p2_values += p2_values[:1]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
            ax.plot(angles, p1_values, 'o-', linewidth=2, label=player1_compare, color='#2196F3')
            ax.fill(angles, p1_values, alpha=0.25, color='#2196F3')
            ax.plot(angles, p2_values, 'o-', linewidth=2, label=player2_compare, color='#FF5722')
            ax.fill(angles, p2_values, alpha=0.25, color='#FF5722')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(['25', '50', '75', '100'], size=8)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            # Dark mode
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['polar'].set_color('white')
            ax.xaxis.label.set_color('white')
            for label in ax.get_xticklabels():
                label.set_color('white')
            for label in ax.get_yticklabels():
                label.set_color('white')
            legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            legend.get_frame().set_facecolor('#2e2e2e')
            for text in legend.get_texts():
                text.set_color('white')
            
            st.pyplot(fig)
        
        with col3:
            st.markdown(f"### {player2_compare}")
            st.metric("ELO", f"{ratings[player2_compare]:.0f}")
            st.metric("Wins", p2_stats["Wins"])
            st.metric("Win %", f"{p2_stats['W/L %']:.1f}%")
            st.metric("Avg Pts Won", p2_stats["Avg Points Won"])
        
        # Head to head
        st.markdown("### 🤜🤛 Head-to-Head Record")
        h2h_matches = [m for m in matches if 
                      (m["player1"] == player1_compare and m["player2"] == player2_compare) or
                      (m["player1"] == player2_compare and m["player2"] == player1_compare)]
        
        if h2h_matches:
            p1_wins = sum(1 for m in h2h_matches if 
                         (m["player1"] == player1_compare and m["score1"] > m["score2"]) or
                         (m["player2"] == player1_compare and m["score2"] > m["score1"]))
            p2_wins = len(h2h_matches) - p1_wins
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{player1_compare} Wins", p1_wins)
            col2.metric("Total Matches", len(h2h_matches))
            col3.metric(f"{player2_compare} Wins", p2_wins)
        else:
            st.info("These players haven't faced each other yet!")


# 1. ELO DISTRIBUTION & TIERS
st.subheader("🎯 ELO Distribution & Player Tiers")

col1, col2 = st.columns([2, 1])

with col1:
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    active_ratings = [ratings[p] for p in active_players]
    
    # Define tier boundaries
    tiers = {
        'Elite': (1100, max(active_ratings) + 50, '#FFD700'),
        'Advanced': (1050, 1100, '#C0C0C0'),
        'Intermediate': (1000, 1050, '#CD7F32'),
        'Beginner': (min(active_ratings) - 50, 1000, '#87CEEB')
    }
    
    # Create histogram
    n, bins, patches = ax.hist(active_ratings, bins=15, edgecolor='black', alpha=0.7)
    
    # Color bars by tier
    for patch, left_edge in zip(patches, bins[:-1]):
        for tier_name, (tier_min, tier_max, color) in tiers.items():
            if tier_min <= left_edge < tier_max:
                patch.set_facecolor(color)
                break
    
    ax.set_xlabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
    ax.set_title('Player Rating Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Dark mode styling
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

with col2:
    st.markdown("### 🏆 Player Tiers")
    
    # Count players in each tier
    tier_counts = {}
    for tier_name, (tier_min, tier_max, color) in tiers.items():
        count = sum(1 for r in active_ratings if tier_min <= r < tier_max)
        tier_counts[tier_name] = count
    
    # Display tier breakdown
    for tier_name in ['Elite', 'Advanced', 'Intermediate', 'Beginner']:
        count = tier_counts[tier_name]
        tier_min, tier_max, color = tiers[tier_name]
        st.markdown(f"**{tier_name}** ({tier_min}-{tier_max}): {count} players")
    
    # Stats
    st.markdown("---")
    st.metric("Median ELO", f"{np.median(active_ratings):.1f}")


# 2. TOP PLAYERS PROGRESSION
st.subheader("🏅 Top Players ELO Progression")

# Get top 5 players by current rating
top_n = min(5, len(active_players))
# Sort active players by their current ratings
top_players_sorted = sorted([(p, ratings[p]) for p in active_players], key=lambda x: x[1], reverse=True)
top_players = [p for p, r in top_players_sorted[:top_n]]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.tab10(np.linspace(0, 1, top_n))

for idx, player in enumerate(top_players):
    player_history = elo_history[player]
    if len(player_history) > 1:
        match_nums = [m for m, r in player_history]
        elos = [r for m, r in player_history]
        ax.plot(match_nums, elos, marker='o', linewidth=2.5, 
                label=player, color=colors[idx], alpha=0.8)

ax.set_xlabel('Match Number', fontsize=12, fontweight='bold')
ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
ax.set_title('Top Players ELO Journey', fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(alpha=0.3)

# Dark mode styling
ax.set_facecolor('#1e1e1e')
fig.patch.set_facecolor('#1e1e1e')
ax.tick_params(colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='best', framealpha=0.9, fontsize=9)
legend.get_frame().set_facecolor('#2e2e2e')
for text in legend.get_texts():
    text.set_color('white')

st.pyplot(fig)


# 3. RECENT FORM INDICATOR
st.subheader("🔥 Recent Form (Last 10 Matches)")

# Calculate recent form for all active players (only those with 10+ matches)
recent_form_data = []
for player in active_players:
    player_matches = [m for m in matches if m["player1"] == player or m["player2"] == player]
    
    # Only include players who have played at least 10 matches
    if len(player_matches) >= 10:
        last_10 = player_matches[-10:]
        
        wins = sum(1 for m in last_10 if (m["player1"] == player and m["score1"] > m["score2"]) or 
                                         (m["player2"] == player and m["score2"] > m["score1"]))
        win_rate = (wins / len(last_10)) * 100
        
        # Form indicator
        if win_rate >= 70:
            form = "🔥 Hot"
        elif win_rate >= 50:
            form = "⚡ Solid"
        elif win_rate >= 30:
            form = "📉 Cooling"
        else:
            form = "🧊 Cold"
        
        recent_form_data.append({
            "Player": player,
            "Last 10 W-L": f"{wins}-{len(last_10)-wins}",
            "Win Rate %": round(win_rate, 1),
            "Form": form
        })

form_df = pd.DataFrame(recent_form_data)
form_df = form_df.sort_values("Win Rate %", ascending=False).reset_index(drop=True)

st.dataframe(form_df, use_container_width=True)


# 5. MATCH COMPETITIVENESS ANALYSIS
st.subheader("⚔️ Match Competitiveness Analysis")

col1, col2 = st.columns(2)

# Calculate score differentials
score_diffs = []
for match in matches:
    diff = abs(match["score1"] - match["score2"])
    score_diffs.append(diff)

with col1:
    # Score differential distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(score_diffs, bins=range(0, max(score_diffs)+2), 
            edgecolor='black', color='#4CAF50', alpha=0.7)
    ax.set_xlabel('Score Differential', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Matches', fontsize=10, fontweight='bold')
    ax.set_title('How Close Are the Matches?', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Dark mode
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

with col2:
    st.markdown("### 📊 Match Stats")
    
    close_matches = sum(1 for d in score_diffs if d <= 2)
    blowouts = sum(1 for d in score_diffs if d >= 5)
    avg_diff = np.mean(score_diffs)
    
    st.metric("Close Matches (≤2 pts)", f"{close_matches} ({close_matches/len(score_diffs)*100:.1f}%)")
    st.metric("Blowouts (≥5 pts)", f"{blowouts} ({blowouts/len(score_diffs)*100:.1f}%)")
    st.metric("Avg Score Differential", f"{avg_diff:.1f} points")


# 7. ACTIVITY & ENGAGEMENT
st.subheader("📅 Activity & Engagement")

# Parse dates and create activity data
from datetime import datetime, timedelta
import calendar

match_dates = []
for match in matches:
    try:
        date_obj = datetime.strptime(match["date"], "%Y-%m-%d")
        match_dates.append(date_obj)
    except:
        continue

if match_dates:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Activity timeline
        from collections import Counter
        date_counts = Counter([d.date() for d in match_dates])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]
        
        ax.bar(dates, counts, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Matches Played', fontsize=10, fontweight='bold')
        ax.set_title('Match Activity Over Time', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Dark mode
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📊 Activity Stats")
        
        # Most active players
        player_match_counts = {}
        for player in active_players:
            count = sum(1 for m in matches if m["player1"] == player or m["player2"] == player)
            player_match_counts[player] = count
        
        top_4_active = sorted(player_match_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        
        st.markdown("**Most Active Players:**")
        for i, (player, count) in enumerate(top_4_active, 1):
            st.markdown(f"{i}. **{player}**: {count} matches")


# 8. PERFORMANCE METRICS DASHBOARD
st.subheader("🎯 Performance Metrics Dashboard")

# Calculate advanced metrics
performance_metrics = []
for player in active_players:
    player_stat = next((s for s in processed_stats if s["Player"] == player), None)
    if not player_stat:
        continue
    
    # Get player's history
    player_history = elo_history[player]
    if len(player_history) < 2:
        continue
    
    elos = [r for m, r in player_history[1:]]  # Exclude initial rating
    current_elo = ratings[player]
    peak_elo = max(elos)
    
    # ELO volatility (consistency)
    elo_std = np.std(elos) if len(elos) > 1 else 0
    consistency_score = max(0, 100 - elo_std)  # Lower std = higher consistency
    
    # Biggest gain/loss
    elo_changes = [elos[i] - elos[i-1] for i in range(1, len(elos))]
    biggest_gain = max(elo_changes) if elo_changes else 0
    biggest_loss = min(elo_changes) if elo_changes else 0
    
    # Giant killer stat (wins against higher-rated opponents)
    giant_killer_wins = 0
    total_underdog_matches = 0
    for match in matches:
        if match["player1"] == player:
            opp = match["player2"]
            # Check if this was an underdog win (opponent had higher ELO at time of match)
            if opp in ratings and match["score1"] > match["score2"]:
                # Simplified: use current ratings as proxy
                if ratings[opp] > current_elo:
                    giant_killer_wins += 1
                    total_underdog_matches += 1
            elif opp in ratings and match["score1"] < match["score2"]:
                if ratings[opp] > current_elo:
                    total_underdog_matches += 1
        elif match["player2"] == player:
            opp = match["player1"]
            if opp in ratings and match["score2"] > match["score1"]:
                if ratings[opp] > current_elo:
                    giant_killer_wins += 1
                    total_underdog_matches += 1
            elif opp in ratings and match["score2"] < match["score1"]:
                if ratings[opp] > current_elo:
                    total_underdog_matches += 1
    
    underdog_rate = (giant_killer_wins / total_underdog_matches * 100) if total_underdog_matches > 0 else 0
    
    performance_metrics.append({
        "Player": player,
        "Current ELO": round(current_elo, 1),
        "Peak ELO": round(peak_elo, 1),
        "ELO vs Peak": round(current_elo - peak_elo, 1),
        "Consistency": round(consistency_score, 1),
        "Biggest Gain": f"+{biggest_gain:.1f}",
        "Biggest Loss": f"{biggest_loss:.1f}",
        "Underdog Wins": giant_killer_wins,
        "Underdog Win %": round(underdog_rate, 1) if total_underdog_matches > 0 else "-"
    })

perf_df = pd.DataFrame(performance_metrics)

# Display in tabs
tab1, tab2, tab3 = st.tabs(["📊 All Metrics", "🏔️ Peak Performance", "🎯 Giant Killers"])

with tab1:
    st.dataframe(perf_df.sort_values("Current ELO", ascending=False), use_container_width=True)

with tab2:
    peak_df = perf_df[["Player", "Current ELO", "Peak ELO", "ELO vs Peak"]].sort_values("Peak ELO", ascending=False)
    st.dataframe(peak_df, use_container_width=True)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_peak = peak_df.head(10)
    x = np.arange(len(top_10_peak))
    width = 0.35
    
    ax.bar(x - width/2, top_10_peak["Current ELO"], width, label='Current ELO', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, top_10_peak["Peak ELO"], width, label='Peak ELO', color='#FFD700', alpha=0.8)
    
    ax.set_xlabel('Player', fontsize=10, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=10, fontweight='bold')
    ax.set_title('Current vs Peak ELO (Top 10)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10_peak["Player"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Dark mode
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend()
    legend.get_frame().set_facecolor('#2e2e2e')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    giant_df = perf_df[perf_df["Underdog Wins"] > 0][["Player", "Underdog Wins", "Underdog Win %"]].sort_values("Underdog Wins", ascending=False)
    if not giant_df.empty:
        st.dataframe(giant_df, use_container_width=True)
        st.info("🎯 'Giant Killers' are players who win against higher-rated opponents!")
    else:
        st.info("No underdog wins recorded yet!")


st.markdown("""
<hr style="border: none; height: 2px; background-color: #FF4B4B; margin: 20px 0;">
""", unsafe_allow_html=True)


# ============================================================
# DOUBLES SECTIONS (MOVED TO BOTTOM)
# ============================================================

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

st.header("📊 Doubles Player Performance Stats")

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
