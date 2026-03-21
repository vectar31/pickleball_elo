# Pickleball Club — Product Spec

**Status:** In development
**Stack:** React Native (Expo) + FastAPI
**Owners:** Swapnil (mobile), Piyush (backend)

---

## Vision

A worldwide cross-platform app where pickleball players create closed communities, invite friends, and track matches — with ELO-based leaderboards, performance comparisons, and community engagement. Think Strava, but for pickleball.

---

## Core Concepts

- **Community** — A closed group of players. Users can belong to multiple communities.
- **Match** — A singles or doubles game recorded by one player, validated by the other within 24 hrs (auto-accepted if no response).
- **ELO** — Each player has an ELO rating per community. Ratings update after each validated match.
- **Leaderboard** — ELO ranking within a community.

---

## First Cut — Screens

### 1. Auth (Login / Sign Up)
- Google SSO
- Apple Sign-In
- No email/password auth in v1

### 2. Onboarding (first login only)
- Display name
- Location (city/country)
- Profile photo (optional)

### 3. Home
- Your communities (horizontal scroll)
- Recent games across all your communities
- Engaging feed content (TBD — e.g. "You climbed 2 spots this week!")

### 4. Search / Discover Communities
- Search by name or code
- Join by invite link or community code
- Create a new community

### 5. Community Page
- Community name, member count
- Recent matches
- ELO leaderboard (ranked list with rating + delta)
- Member list

### 6. Add Game
- Select community
- Select format: singles or doubles
- Select players from community members
- Enter score (per-round support)
- Submit → other player gets notified to confirm
- If not confirmed within 24 hrs → auto-accepted

### 7. Profile
- Display name, location, photo
- ELO ratings per community
- Match history
- Win/loss stats

---

## Maker-Checker Validation

When a match is submitted:
1. Status = `pending_validation`
2. The opposing player is notified (push + in-app)
3. They can **confirm** or **dispute**
4. If no action within 24 hrs → status = `accepted` automatically
5. Disputed matches are flagged for community admin review (v2)

---

## ELO Scoring

- Each player starts at 1000 ELO per community
- K-factor: 32 (standard)
- ELO recalculated after each accepted match
- Doubles: team ELO calculated as average of team members

---

## Out of Scope (v1)

- Tournament brackets
- Video/media uploads
- In-app chat
- Admin dispute resolution UI
- Public (non-community) leaderboards
- Web app
