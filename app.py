import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from elo import load_players, compute_ratings_and_history, compute_doubles_ratings_and_history, compute_ratings_from_data

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pickleball Elo Ratings", layout="wide", initial_sidebar_state="collapsed")

ACCENT  = "#67cfff"
GREEN   = "#4CAF50"
GOLD    = "#FFD700"
RED     = "#FF5252"

# ── Data Loading ──────────────────────────────────────────────────────────────
ratings, history, matches = compute_ratings_and_history()
doubles_ratings, doubles_history, doubles_matches = compute_doubles_ratings_and_history()
elo_history = history

active_players = set()
for m in matches:
    active_players.update([m["player1"], m["player2"]])

all_registered = sorted(load_players())

# ── Previous-session ratings (for rank-change arrows) ─────────────────────────
if matches:
    last_date = max(m["date"] for m in matches)
    prev_matches = [m for m in matches if m["date"] < last_date]
    if prev_matches:
        prev_ratings, _ = compute_ratings_from_data(prev_matches, list(load_players()))
    else:
        prev_ratings = {p: 1000 for p in active_players}

    curr_ranked = sorted(active_players, key=lambda p: -ratings[p])
    prev_ranked = sorted(active_players, key=lambda p: -prev_ratings.get(p, 1000))
    curr_rank   = {p: i + 1 for i, p in enumerate(curr_ranked)}
    prev_rank   = {p: i + 1 for i, p in enumerate(prev_ranked)}
    rank_change = {p: prev_rank.get(p, len(active_players)) - curr_rank[p] for p in active_players}
else:
    rank_change = {}

# ── This Week's Champion ──────────────────────────────────────────────────────
today    = datetime.today().date()
week_ago = today - timedelta(days=7)

weekly_wins    = defaultdict(int)
weekly_matches = defaultdict(int)
for m in matches:
    try:
        md = datetime.strptime(m["date"], "%Y-%m-%d").date()
    except Exception:
        continue
    if md >= week_ago:
        p1, p2 = m["player1"], m["player2"]
        weekly_matches[p1] += 1
        weekly_matches[p2] += 1
        if m["score1"] > m["score2"]:
            weekly_wins[p1] += 1
        else:
            weekly_wins[p2] += 1

weekly_eligible = {
    p: weekly_wins[p] / weekly_matches[p]
    for p in weekly_matches if weekly_matches[p] >= 3
}
champ_this_week = max(weekly_eligible, key=weekly_eligible.get) if weekly_eligible else None

# ── Build interpolated Elo history DataFrame ──────────────────────────────────
max_match_num = max((mn for series in elo_history.values() for mn, _ in series), default=0)

graph_data = []
for player, series in elo_history.items():
    if player not in active_players or not series:
        continue
    for i, (mn, rating) in enumerate(series):
        graph_data.append({"Player": player, "Match #": mn, "Elo Rating": rating})
        if i < len(series) - 1:
            next_mn = series[i + 1][0]
            for skipped in range(mn + 1, next_mn):
                graph_data.append({"Player": player, "Match #": skipped, "Elo Rating": rating})
    last_mn, last_rating = series[-1]
    if last_mn < max_match_num:
        graph_data.append({"Player": player, "Match #": max_match_num, "Elo Rating": last_rating})

graph_df = pd.DataFrame(graph_data)

# ── Singles stats computation ─────────────────────────────────────────────────
def max_streak(seq, target):
    mx = cnt = 0
    for r in seq:
        cnt = cnt + 1 if r == target else 0
        mx  = max(mx, cnt)
    return mx

raw_stats = defaultdict(lambda: {
    "Wins": 0, "Losses": 0, "Games": 0,
    "Points Won": 0, "Points Lost": 0, "Streak History": []
})
for m in matches:
    p1, p2, s1, s2 = m["player1"], m["player2"], m["score1"], m["score2"]
    winner = p1 if s1 > s2 else p2
    loser  = p2 if s1 > s2 else p1
    raw_stats[winner]["Wins"]   += 1
    raw_stats[loser]["Losses"]  += 1
    raw_stats[p1]["Points Won"]  += s1;  raw_stats[p1]["Points Lost"] += s2
    raw_stats[p2]["Points Won"]  += s2;  raw_stats[p2]["Points Lost"] += s1
    raw_stats[p1]["Games"] += 1;  raw_stats[p2]["Games"] += 1
    for player in [p1, p2]:
        raw_stats[player]["Streak History"].append("W" if player == winner else "L")

processed_stats = []
for player, data in raw_stats.items():
    if player not in active_players:
        continue
    games  = data["Games"]
    wins   = data["Wins"]
    losses = data["Losses"]
    seq    = data["Streak History"]
    streak = ""
    if seq:
        last  = seq[-1]
        cnt   = sum(1 for _ in (x for x in reversed(seq) if x == last))
        # recount properly
        cnt = 0
        for r in reversed(seq):
            if r == last: cnt += 1
            else:         break
        streak = f"{cnt}{last}"
    processed_stats.append({
        "Player":              player,
        "Matches":             games,
        "Wins":                wins,
        "Losses":              losses,
        "W/L %":               round(wins / games * 100, 1) if games else 0,
        "Current Streak":      streak,
        "Longest Win Streak":  max_streak(seq, "W"),
        "Longest Loss Streak": max_streak(seq, "L"),
        "Avg Points Won":      round(data["Points Won"]  / games, 1) if games else 0,
        "Avg Points Lost":     round(data["Points Lost"] / games, 1) if games else 0,
    })

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("📊 Club Stats")
doubles_players = set()
for m in doubles_matches:
    doubles_players.update(m["team1"] + m["team2"])

st.sidebar.metric("Total Members",      len(set(all_registered)))
st.sidebar.metric("Singles Players",    len(active_players))
st.sidebar.metric("Doubles Players",    len(doubles_players))
st.sidebar.metric("Singles Matches",    len(matches))
st.sidebar.metric("Doubles Matches",    len(doubles_matches))
st.sidebar.metric("Total Matches",      len(matches) + len(doubles_matches))

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏓 Pickleball Elo Tracker")

if champ_this_week:
    wins_w   = weekly_wins[champ_this_week]
    played_w = weekly_matches[champ_this_week]
    wr_w     = round(weekly_eligible[champ_this_week] * 100)
    st.success(
        f"🏆 **This Week's Champion: {champ_this_week}** — "
        f"{wins_w}W / {played_w - wins_w}L ({wr_w}% win rate this week)"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SINGLES LEADERBOARD (with rank change)
# ═══════════════════════════════════════════════════════════════════════════════
st.header("🏆 Singles Elo Rankings")

leaderboard_rows = []
for i, player in enumerate(sorted(active_players, key=lambda p: -ratings[p]), 1):
    delta = rank_change.get(player, 0)
    arrow = f"↑{delta}" if delta > 0 else (f"↓{abs(delta)}" if delta < 0 else "—")
    ps    = next((s for s in processed_stats if s["Player"] == player), {})
    leaderboard_rows.append({
        "Rank":           i,
        "Player":         player,
        "ELO":            round(ratings[player]),
        "Rank Change":    arrow,
        "Matches":        ps.get("Matches", 0),
        "Win %":          ps.get("W/L %", 0),
        "Streak":         ps.get("Current Streak", ""),
    })

st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE-PLAYER ELO JOURNEY (Plotly, with peak annotation)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Singles Elo History")

unique_singles_players = sorted(graph_df["Player"].unique())
_default_single = unique_singles_players.index("Arora") if "Arora" in unique_singles_players else 0
selected_single = st.selectbox("Select a player:", unique_singles_players, index=_default_single, key="singles_player_smooth")

spdf = graph_df[graph_df["Player"] == selected_single].sort_values("Match #")
actual_matches = spdf[spdf["Elo Rating"] != spdf["Elo Rating"].shift()].reset_index(drop=True)
actual_matches["Player Match #"] = actual_matches.index + 1
spdf = spdf.merge(actual_matches[["Match #", "Player Match #"]], on="Match #", how="left")
spdf["Player Match #"] = spdf["Player Match #"].ffill()

elo_vals  = actual_matches["Elo Rating"].values
match_nos = actual_matches["Player Match #"].values

annotations = []
if len(elo_vals) > 0:
    peak_idx = int(np.argmax(elo_vals))
    annotations.append(dict(
        x=float(match_nos[peak_idx]), y=float(elo_vals[peak_idx]),
        text=f"🏔 Peak {elo_vals[peak_idx]:.0f}",
        showarrow=True, arrowhead=2,
        bgcolor=GOLD, font=dict(color="black", size=11),
        bordercolor=GOLD
    ))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=spdf["Player Match #"], y=spdf["Elo Rating"],
    mode="lines+markers",
    line=dict(color=ACCENT, width=2),
    marker=dict(size=4),
    name="Elo",
    hovertemplate="Match %{x}<br>ELO: %{y:.0f}<extra></extra>"
))
_elo_min = float(spdf["Elo Rating"].min())
_elo_max = float(spdf["Elo Rating"].max())
_elo_buf = max(15, (_elo_max - _elo_min) * 0.08)
fig.update_layout(
    title=f"📈 Elo Journey: {selected_single}",
    xaxis_title="Player's Match #", yaxis_title="Elo Rating",
    yaxis=dict(range=[_elo_min - _elo_buf, _elo_max + _elo_buf]),
    template="plotly_dark", annotations=annotations, hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOP PLAYERS ELO PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🏅 Top Players ELO Progression")

top_n       = min(5, len(active_players))
top_players = [p for p, _ in sorted([(p, ratings[p]) for p in active_players], key=lambda x: -x[1])[:top_n]]
colors_seq  = px.colors.qualitative.Set2

fig = go.Figure()
for idx, player in enumerate(top_players):
    ph = elo_history[player]
    fig.add_trace(go.Scatter(
        x=[mn for mn, _ in ph], y=[r for _, r in ph],
        mode="lines+markers", name=player,
        line=dict(color=colors_seq[idx % len(colors_seq)], width=2.5),
        marker=dict(size=5),
        hovertemplate=f"{player}<br>Match %{{x}}<br>ELO: %{{y:.0f}}<extra></extra>"
    ))
fig.update_layout(
    title="Top Players ELO Journey", xaxis_title="Match #", yaxis_title="ELO",
    template="plotly_dark", hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. HEAD-TO-HEAD DOMINANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Head-to-Head Dominance Matrix")
st.caption("Win % of the row player against the column player. Green = dominates, Red = struggles.")

# Limit to players with ≥ 5 matches for readability
matrix_players = sorted([
    p for p in active_players
    if sum(1 for m in matches if m["player1"] == p or m["player2"] == p) >= 5
])

z_vals     = []
hover_text = []

for p1 in matrix_players:
    row_z, row_h = [], []
    for p2 in matrix_players:
        if p1 == p2:
            row_z.append(None)
            row_h.append("")
        else:
            h2h = [
                m for m in matches
                if (m["player1"] == p1 and m["player2"] == p2) or
                   (m["player1"] == p2 and m["player2"] == p1)
            ]
            if not h2h:
                row_z.append(None)
                row_h.append(f"{p1} vs {p2}<br>No matches yet")
            else:
                wins = sum(
                    1 for m in h2h
                    if (m["player1"] == p1 and m["score1"] > m["score2"]) or
                       (m["player2"] == p1 and m["score2"] > m["score1"])
                )
                pct = round(wins / len(h2h) * 100, 1)
                row_z.append(pct)
                row_h.append(f"{p1} vs {p2}<br>Win%: {pct:.0f}%<br>Matches: {len(h2h)}")
    z_vals.append(row_z)
    hover_text.append(row_h)

fig = go.Figure(data=go.Heatmap(
    z=z_vals, x=matrix_players, y=matrix_players,
    colorscale="RdYlGn", zmid=50,
    text=[[f"{v:.0f}%" if v is not None else "" for v in row] for row in z_vals],
    texttemplate="%{text}",
    hovertext=hover_text, hoverinfo="text",
    showscale=True, colorbar=dict(title="Win %"),
    zmin=0, zmax=100,
))
fig.update_layout(
    title="H2H Win % (row beats column)",
    template="plotly_dark",
    xaxis=dict(tickangle=45),
    height=max(400, len(matrix_players) * 28)
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ELO BAR CHART RACE (animated)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🏎️ ELO Bar Chart Race")
st.caption("Hit ▶ Play to watch the ELO standings evolve match by match.")

race_players = [
    p for p, _ in sorted(
        [(p, ratings[p]) for p in active_players if len(elo_history[p]) > 2],
        key=lambda x: -x[1]
    )[:12]
]

palette      = px.colors.qualitative.Set3
colors_race  = {p: palette[i % len(palette)] for i, p in enumerate(race_players)}

def elo_at(player, match_num):
    elo = 1000.0
    for mn, r in elo_history[player]:
        if mn <= match_num:
            elo = r
        else:
            break
    return elo

# Sample frames (every 2 matches, always include final)
step       = max(1, max_match_num // 60)
frame_nums = list(range(0, max_match_num + 1, step))
if max_match_num not in frame_nums:
    frame_nums.append(max_match_num)

x_min = min(800, min(ratings.values()) - 50)
x_max = max(ratings.values()) + 120

def make_bar_data(fn):
    fd = sorted([(p, elo_at(p, fn)) for p in race_players], key=lambda x: x[1])
    return [d[0] for d in fd], [d[1] for d in fd]

init_players, init_elos = make_bar_data(0)

frames = []
for fn in frame_nums:
    fp, fe = make_bar_data(fn)
    frames.append(go.Frame(
        data=[go.Bar(
            x=fe, y=fp, orientation="h",
            marker_color=[colors_race[p] for p in fp],
            text=[f"{e:.0f}" for e in fe], textposition="outside",
        )],
        name=str(fn),
        layout=go.Layout(title_text=f"ELO Standings — Match #{fn}")
    ))

fig = go.Figure(
    data=[go.Bar(
        x=init_elos, y=init_players, orientation="h",
        marker_color=[colors_race[p] for p in init_players],
        text=[f"{e:.0f}" for e in init_elos], textposition="outside",
    )],
    frames=frames,
    layout=go.Layout(
        title="ELO Bar Chart Race",
        template="plotly_dark",
        height=500,
        xaxis=dict(range=[x_min, x_max], title="ELO", fixedrange=True),
        yaxis=dict(fixedrange=True),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.18, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",  method="animate",
                     args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                method="animate",
                args=[[str(fn)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=str(fn)
            ) for fn in frame_nums],
            x=0.05, y=0, xanchor="left", yanchor="top", len=0.95,
            currentvalue=dict(prefix="Match: ", font=dict(size=13)),
        )]
    )
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLAYER COMPARISON TOOL
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔬 Player Comparison Tool")

sorted_active = sorted(active_players)
default_p1 = "Arora" if "Arora" in active_players else sorted_active[0]
default_p2 = "Ameta" if "Ameta" in active_players else (sorted_active[1] if len(sorted_active) > 1 else sorted_active[0])

col1, col2 = st.columns(2)
with col1:
    p1_sel = st.selectbox("Player 1:", sorted_active, index=sorted_active.index(default_p1), key="compare1")
with col2:
    p2_sel = st.selectbox("Player 2:", sorted_active, index=sorted_active.index(default_p2), key="compare2")

if p1_sel != p2_sel:
    p1s = next((s for s in processed_stats if s["Player"] == p1_sel), None)
    p2s = next((s for s in processed_stats if s["Player"] == p2_sel), None)

    if p1s and p2s:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(f"### {p1_sel}")
            st.metric("ELO",    f"{ratings[p1_sel]:.0f}")
            st.metric("Wins",   p1s["Wins"])
            st.metric("Win %",  f"{p1s['W/L %']:.1f}%")
            st.metric("Matches", p1s["Matches"])

        with col3:
            st.markdown(f"### {p2_sel}")
            st.metric("ELO",    f"{ratings[p2_sel]:.0f}")
            st.metric("Wins",   p2s["Wins"])
            st.metric("Win %",  f"{p2s['W/L %']:.1f}%")
            st.metric("Matches", p2s["Matches"])

        with col2:
            min_elo = min(ratings.values())
            max_elo = max(ratings.values())
            elo_range = max_elo - min_elo or 1

            def norm(v, mn, mx): return (v - mn) / (mx - mn) * 100 if mx > mn else 50

            cats   = ["ELO", "Win Rate", "Avg Pts Won", "Longest Win Streak", "Matches Played"]
            p1_vals = [
                norm(ratings[p1_sel], min_elo, max_elo),
                p1s["W/L %"],
                (p1s["Avg Points Won"] / 11) * 100,
                min(p1s["Longest Win Streak"] / 10 * 100, 100),
                min(p1s["Matches"] / 50 * 100, 100),
            ]
            p2_vals = [
                norm(ratings[p2_sel], min_elo, max_elo),
                p2s["W/L %"],
                (p2s["Avg Points Won"] / 11) * 100,
                min(p2s["Longest Win Streak"] / 10 * 100, 100),
                min(p2s["Matches"] / 50 * 100, 100),
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=p1_vals, theta=cats, fill="toself",
                name=p1_sel, line_color="#2196F3"
            ))
            fig.add_trace(go.Scatterpolar(
                r=p2_vals, theta=cats, fill="toself",
                name=p2_sel, line_color="#FF5722"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                template="plotly_dark", showlegend=True,
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        # H2H record
        st.markdown("### 🤜🤛 Head-to-Head Record")
        h2h = [
            m for m in matches
            if (m["player1"] == p1_sel and m["player2"] == p2_sel) or
               (m["player1"] == p2_sel and m["player2"] == p1_sel)
        ]
        if h2h:
            p1_wins = sum(
                1 for m in h2h
                if (m["player1"] == p1_sel and m["score1"] > m["score2"]) or
                   (m["player2"] == p1_sel and m["score2"] > m["score1"])
            )
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{p1_sel} Wins", p1_wins)
            c2.metric("Total Matches",  len(h2h))
            c3.metric(f"{p2_sel} Wins", len(h2h) - p1_wins)
        else:
            st.info("These players haven't faced each other yet!")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RECENT FORM
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔥 Recent Form (Last 10 Matches)")

recent_form_data = []
for player in active_players:
    pm = [m for m in matches if m["player1"] == player or m["player2"] == player]
    if len(pm) >= 10:
        last_10 = pm[-10:]
        wins = sum(
            1 for m in last_10
            if (m["player1"] == player and m["score1"] > m["score2"]) or
               (m["player2"] == player and m["score2"] > m["score1"])
        )
        wr = wins / len(last_10) * 100
        form = "🔥 Hot" if wr >= 70 else "⚡ Solid" if wr >= 50 else "📉 Cooling" if wr >= 30 else "🧊 Cold"
        recent_form_data.append({
            "Player":       player,
            "Last 10 W-L":  f"{wins}-{len(last_10)-wins}",
            "Win Rate %":   round(wr, 1),
            "Form":         form,
        })

if recent_form_data:
    form_df = pd.DataFrame(recent_form_data).sort_values("Win Rate %", ascending=False).reset_index(drop=True)
    st.dataframe(form_df, use_container_width=True, hide_index=True)
else:
    st.info("Not enough data yet (need players with ≥10 matches).")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE METRICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Performance Metrics Dashboard")

perf_rows = []
for player in active_players:
    ph = elo_history[player]
    if len(ph) < 2:
        continue
    elos    = [r for _, r in ph[1:]]
    changes = [elos[i] - elos[i - 1] for i in range(1, len(elos))]
    perf_rows.append({
        "Player":      player,
        "Current ELO": round(ratings[player], 1),
        "Peak ELO":    round(max(elos), 1),
        "vs Peak":     round(ratings[player] - max(elos), 1),
        "Consistency": round(max(0.0, 100 - float(np.std(elos))), 1),
        "Biggest Gain": f"+{max(changes):.1f}" if changes else "—",
        "Biggest Loss": f"{min(changes):.1f}"  if changes else "—",
    })

perf_df = pd.DataFrame(perf_rows)
tab1, tab2 = st.tabs(["📊 All Metrics", "🏔️ Current vs Peak ELO"])

with tab1:
    st.dataframe(
        perf_df.sort_values("Current ELO", ascending=False),
        use_container_width=True, hide_index=True
    )

with tab2:
    top10 = perf_df.sort_values("Peak ELO", ascending=False).head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top10["Player"], y=top10["Current ELO"], name="Current ELO", marker_color="#2196F3"))
    fig.add_trace(go.Bar(x=top10["Player"], y=top10["Peak ELO"],    name="Peak ELO",    marker_color=GOLD))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        title="Current vs Peak ELO (Top 10)", xaxis_tickangle=45
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. MATCH COMPETITIVENESS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("⚔️ Match Competitiveness")

score_diffs = [abs(m["score1"] - m["score2"]) for m in matches]
col1, col2  = st.columns([2, 1])

with col1:
    fig = px.histogram(
        x=score_diffs,
        nbins=max(score_diffs) - min(score_diffs) + 1 if score_diffs else 10,
        color_discrete_sequence=[GREEN],
        template="plotly_dark",
        title="Score Differential Distribution",
        labels={"x": "Score Differential", "y": "Matches"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    close    = sum(1 for d in score_diffs if d <= 2)
    blowouts = sum(1 for d in score_diffs if d >= 5)
    st.metric("Close Matches (≤2 pts)", f"{close} ({close/len(score_diffs)*100:.1f}%)")
    st.metric("Blowouts (≥5 pts)",      f"{blowouts} ({blowouts/len(score_diffs)*100:.1f}%)")
    st.metric("Avg Score Diff",         f"{np.mean(score_diffs):.1f} pts")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. SCORING PROFILE (Points Map)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Scoring Profile")
st.caption(
    "**Avg points conceded in wins** — lower means more dominant victories. "
    "**Avg score in losses** — higher means more competitive even when losing. "
    "Sorted by ELO."
)

pts_rows = []
for player in active_players:
    conceded_in_wins  = []   # opponent's score when this player wins
    scored_in_losses  = []   # this player's score when they lose
    for m in matches:
        if m["player1"] == player:
            if m["score1"] > m["score2"]:
                conceded_in_wins.append(m["score2"])   # opponent score
            else:
                scored_in_losses.append(m["score1"])   # own score in loss
        elif m["player2"] == player:
            if m["score2"] > m["score1"]:
                conceded_in_wins.append(m["score1"])   # opponent score
            else:
                scored_in_losses.append(m["score2"])   # own score in loss
    if conceded_in_wins or scored_in_losses:
        pts_rows.append({
            "Player":                   player,
            "Avg Conceded in Wins":     round(float(np.mean(conceded_in_wins)),  1) if conceded_in_wins  else None,
            "Avg Score in Losses":      round(float(np.mean(scored_in_losses)), 1) if scored_in_losses else None,
            "ELO":                      ratings[player],
        })

pts_df = pd.DataFrame(pts_rows).sort_values("ELO", ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=pts_df["Player"], y=pts_df["Avg Conceded in Wins"],
    name="Avg Conceded in Wins (lower = more dominant)",
    marker_color="#FF9800"
))
fig.add_trace(go.Bar(
    x=pts_df["Player"], y=pts_df["Avg Score in Losses"],
    name="Avg Score in Losses (higher = more competitive)",
    marker_color="#2196F3"
))
fig.update_layout(
    barmode="group", template="plotly_dark",
    title="Scoring Profile (sorted by ELO)", xaxis_tickangle=45,
    yaxis_title="Points"
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 12. CLUB RIVALRY NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🕸️ Club Rivalry Network")
st.caption("Node size & color = ELO. Edge thickness = matches played between the two.")

h2h_pairs = defaultdict(lambda: {"matches": 0})
for m in matches:
    key = tuple(sorted([m["player1"], m["player2"]]))
    h2h_pairs[key]["matches"] += 1

net_players = list(active_players)
n           = len(net_players)
angles      = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos         = {p: (float(np.cos(a)), float(np.sin(a))) for p, a in zip(net_players, angles)}

edge_traces = []
for (p1, p2), data in h2h_pairs.items():
    if p1 not in pos or p2 not in pos:
        continue
    x0, y0 = pos[p1]
    x1, y1 = pos[p2]
    width   = min(1 + data["matches"] * 0.6, 7)
    edge_traces.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(width=width, color="rgba(103,207,255,0.25)"),
        hoverinfo="skip"
    ))

min_r = min(ratings.values())
max_r = max(ratings.values())
r_range = max_r - min_r or 1

node_trace = go.Scatter(
    x=[pos[p][0] for p in net_players],
    y=[pos[p][1] for p in net_players],
    mode="markers+text",
    text=net_players,
    textposition="top center",
    marker=dict(
        size=[12 + (ratings[p] - min_r) / r_range * 28 for p in net_players],
        color=[ratings[p] for p in net_players],
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="ELO"),
    ),
    hovertext=[f"{p}<br>ELO: {ratings[p]:.0f}" for p in net_players],
    hoverinfo="text"
)

fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(
    title="Club Rivalry Network",
    template="plotly_dark", showlegend=False, height=650,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 13. ELO DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 ELO Distribution & Tiers")

active_ratings_list = [ratings[p] for p in active_players]
tiers = {
    "Elite":        (1100, float("inf"), GOLD),
    "Advanced":     (1050, 1100,         "#C0C0C0"),
    "Intermediate": (1000, 1050,         "#CD7F32"),
    "Beginner":     (0,    1000,         "#87CEEB"),
}

col1, col2 = st.columns([2, 1])
with col1:
    fig = px.histogram(
        x=active_ratings_list, nbins=15,
        color_discrete_sequence=[ACCENT],
        template="plotly_dark",
        title="Player Rating Distribution",
        labels={"x": "ELO Rating", "y": "Players"},
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.markdown("### 🏆 Player Tiers")
    for tier_name, (lo, hi, _) in tiers.items():
        cnt = sum(1 for r in active_ratings_list if lo <= r < hi)
        label = f"{lo}+" if hi == float("inf") else f"{lo}–{hi}"
        st.markdown(f"**{tier_name}** ({label}): {cnt} players")
    st.markdown("---")
    st.metric("Median ELO", f"{np.median(active_ratings_list):.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. ACTIVITY & ENGAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Activity Over Time")

match_dates = []
for m in matches:
    try:
        match_dates.append(datetime.strptime(m["date"], "%Y-%m-%d").date())
    except Exception:
        continue

if match_dates:
    date_counts = Counter(match_dates)
    dates_sorted = sorted(date_counts.keys())

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            x=dates_sorted, y=[date_counts[d] for d in dates_sorted],
            labels={"x": "Date", "y": "Matches"},
            color_discrete_sequence=[GREEN],
            template="plotly_dark",
            title="Match Activity Over Time",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Most Active Players")
        pm_counts = {p: sum(1 for m in matches if m["player1"] == p or m["player2"] == p) for p in active_players}
        for i, (p, cnt) in enumerate(sorted(pm_counts.items(), key=lambda x: -x[1])[:5], 1):
            st.markdown(f"{i}. **{p}**: {cnt} matches")

# ═══════════════════════════════════════════════════════════════════════════════
# 15. PLAYER STATS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Full Player Stats")
stats_df = pd.DataFrame(processed_stats).sort_values("Wins", ascending=False).reset_index(drop=True)
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# 16. Match History (collapsed)
with st.expander("📜 Singles Match History", expanded=False):
    match_df = pd.DataFrame(matches)
    if not match_df.empty:
        st.dataframe(match_df[::-1], use_container_width=True)
    else:
        st.write("No matches yet.")

# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLES SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""<hr style="border:none;height:2px;background:#FF4B4B;margin:30px 0;">""", unsafe_allow_html=True)
st.header("👯 Doubles")

# Doubles leaderboard
st.subheader("🏆 Doubles ELO Rankings")
doubles_data = [
    {"Player": p, "Doubles ELO": round(r, 1)}
    for p, r in doubles_ratings.items()
    if len(doubles_history.get(p, [])) > 1
]
if doubles_data:
    df_d = pd.DataFrame(doubles_data).sort_values("Doubles ELO", ascending=False).reset_index(drop=True)
    st.dataframe(df_d, use_container_width=True, hide_index=True)
else:
    st.info("No doubles matches played yet.")

# Doubles Elo history — build graph df
max_dm = max((mn for series in doubles_history.values() for mn, _ in series), default=0)
dbl_graph_data = []
for player, series in doubles_history.items():
    if not series:
        continue
    for i, (mn, rating) in enumerate(series):
        dbl_graph_data.append({"Player": player, "Match #": mn, "Doubles ELO": rating})
        if i < len(series) - 1:
            for s in range(mn + 1, series[i + 1][0]):
                dbl_graph_data.append({"Player": player, "Match #": s, "Doubles ELO": rating})
    last_mn, last_r = series[-1]
    for extra in range(last_mn + 1, max_dm + 1):
        dbl_graph_data.append({"Player": player, "Match #": extra, "Doubles ELO": last_r})

if dbl_graph_data:
    dbl_graph_df = pd.DataFrame(dbl_graph_data)

    st.subheader("🔍 Doubles Elo History")
    unique_dbl = sorted(dbl_graph_df["Player"].unique())
    sel_dbl    = st.selectbox("Select a player:", unique_dbl, key="doubles_player_smooth")

    pdf = dbl_graph_df[dbl_graph_df["Player"] == sel_dbl].sort_values("Match #")
    actual_dbl = pdf[pdf["Doubles ELO"] != pdf["Doubles ELO"].shift()].reset_index(drop=True)
    actual_dbl["Player Match #"] = actual_dbl.index + 1
    pdf = pdf.merge(actual_dbl[["Match #", "Player Match #"]], on="Match #", how="left")
    pdf["Player Match #"] = pdf["Player Match #"].ffill()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf["Player Match #"], y=pdf["Doubles ELO"],
        mode="lines",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="Match %{x}<br>ELO: %{y:.0f}<extra></extra>"
    ))
    _dbl_min = float(pdf["Doubles ELO"].min())
    _dbl_max = float(pdf["Doubles ELO"].max())
    _dbl_buf = max(15, (_dbl_max - _dbl_min) * 0.08)
    fig.update_layout(
        title=f"📈 Doubles Elo: {sel_dbl}",
        xaxis_title="Match #", yaxis_title="Doubles ELO",
        yaxis=dict(range=[_dbl_min - _dbl_buf, _dbl_max + _dbl_buf]),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# Doubles stats
st.subheader("📊 Doubles Player Stats")
dbl_raw = defaultdict(lambda: {"Wins": 0, "Losses": 0, "Games": 0, "Points Won": 0, "Points Lost": 0, "Streak History": []})
for m in doubles_matches:
    t1, t2, s1, s2 = m["team1"], m["team2"], m["score1"], m["score2"]
    winners = t1 if s1 > s2 else t2
    losers  = t2 if s1 > s2 else t1
    for p in winners:
        dbl_raw[p]["Wins"] += 1;  dbl_raw[p]["Streak History"].append("W")
    for p in losers:
        dbl_raw[p]["Losses"] += 1;  dbl_raw[p]["Streak History"].append("L")
    for p in t1 + t2:
        dbl_raw[p]["Games"] += 1
    for p in t1:
        dbl_raw[p]["Points Won"] += s1;  dbl_raw[p]["Points Lost"] += s2
    for p in t2:
        dbl_raw[p]["Points Won"] += s2;  dbl_raw[p]["Points Lost"] += s1

proc_dbl = []
for player, data in dbl_raw.items():
    if player not in doubles_ratings:
        continue
    games = data["Games"]
    wins  = data["Wins"]
    seq   = data["Streak History"]
    streak = ""
    if seq:
        last = seq[-1]; cnt = 0
        for r in reversed(seq):
            if r == last: cnt += 1
            else:         break
        streak = f"{cnt}{last}"
    proc_dbl.append({
        "Player":              player,
        "Matches":             games,
        "Wins":                wins,
        "Losses":              data["Losses"],
        "W/L %":               round(wins / games * 100, 1) if games else 0,
        "Current Streak":      streak,
        "Longest Win Streak":  max_streak(seq, "W"),
        "Longest Loss Streak": max_streak(seq, "L"),
        "Avg Pts Won":         round(data["Points Won"]  / games, 1) if games else 0,
        "Avg Pts Lost":        round(data["Points Lost"] / games, 1) if games else 0,
    })

if proc_dbl:
    dbl_stats_df = pd.DataFrame(proc_dbl).sort_values("Wins", ascending=False).reset_index(drop=True)
    st.dataframe(dbl_stats_df, use_container_width=True, hide_index=True)

# Doubles partner synergy
st.header("🤝 Doubles Partner Synergy & Matchup Stats")
partner_stats  = defaultdict(lambda: defaultdict(lambda: {"matches": 0, "wins": 0}))
matchup_stats  = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0}))

for m in doubles_matches:
    t1, t2, s1, s2 = m["team1"], m["team2"], m["score1"], m["score2"]
    winning_team = t1 if s1 > s2 else t2
    for team in [t1, t2]:
        for pa in team:
            for pb in team:
                if pa != pb:
                    partner_stats[pa][pb]["matches"] += 1
                    if team == winning_team:
                        partner_stats[pa][pb]["wins"] += 1
    k1 = " & ".join(sorted(t1))
    k2 = " & ".join(sorted(t2))
    if s1 > s2:
        matchup_stats[k1][k2]["wins"]   += 1;  matchup_stats[k1][k2]["total"] += 1
        matchup_stats[k2][k1]["losses"] += 1;  matchup_stats[k2][k1]["total"] += 1
    else:
        matchup_stats[k2][k1]["wins"]   += 1;  matchup_stats[k2][k1]["total"] += 1
        matchup_stats[k1][k2]["losses"] += 1;  matchup_stats[k1][k2]["total"] += 1

bp_rows = []
for player, partners in partner_stats.items():
    best_p, best_s = max(partners.items(), key=lambda x: x[1]["wins"], default=(None, {"wins": 0, "matches": 0}))
    if best_p:
        total = best_s["matches"]
        wins  = best_s["wins"]
        bp_rows.append({
            "Player":          player,
            "Best Partner":    best_p,
            "Matches Together": total,
            "Wins Together":   wins,
            "Win %":           round(100 * wins / total, 1) if total else 0,
        })

st.subheader("🏅 Best Doubles Partner")
if bp_rows:
    st.dataframe(pd.DataFrame(bp_rows).sort_values("Win %", ascending=False), use_container_width=True, hide_index=True)

mu_rows = []
for k1, opps in matchup_stats.items():
    for k2, ms in opps.items():
        if k1 < k2:
            mu_rows.append({
                "Team 1": k1, "Team 2": k2,
                "Wins": ms["wins"], "Losses": ms["losses"],
                "Total": ms["total"],
                "Win %": round(100 * ms["wins"] / ms["total"], 1) if ms["total"] else 0,
            })

st.subheader("🏓 Doubles Matchup Records")
if mu_rows:
    st.dataframe(
        pd.DataFrame(mu_rows).sort_values("Total", ascending=False),
        use_container_width=True, hide_index=True
    )

with st.expander("📜 Doubles Match History", expanded=False):
    if doubles_matches:
        dbl_mdf = pd.DataFrame([{
            "Date":    m["date"],
            "Team 1":  " + ".join(m["team1"]),
            "Score 1": m["score1"],
            "Score 2": m["score2"],
            "Team 2":  " + ".join(m["team2"]),
        } for m in doubles_matches])
        st.dataframe(dbl_mdf[::-1], use_container_width=True)
    else:
        st.write("No doubles matches yet.")

with st.expander("🧑‍🤝‍🧑 Club Members", expanded=False):
    members_df = pd.DataFrame([{
        "Player": p,
        "Status": "🟢 Rated" if p in active_players else "⚪ Unrated"
    } for p in all_registered])
    st.dataframe(members_df, use_container_width=True, hide_index=True)
