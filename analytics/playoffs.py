"""
Playoff picture: division leaders, the wild card race, a bracket, and a
projected playoff roster - v1 first pass, built to be adjusted.

Division leaders / wild card race are computed live off
analytics/standings.py's own compute_standings() (the same real win/loss
records the Standings page itself uses) - no separate data source.

BRACKET: two real, distinct data paths, not one projection stretched to
cover both:
  1. REAL RESULTS (real_postseason_bracket / _fetch_postseason_games) -
     the MLB Stats API's own schedule endpoint, filtered to postseason
     game types (F=Wild Card, D=Division Series, L=Championship Series,
     W=World Series), the same officially-sourced live-API pattern this
     project already uses for ERA (analytics/boxscore.py). Used whenever
     ANY real "Final" postseason game exists for the requested season -
     covers every completed past season AND the current season once real
     playoff games start being played, filling in whichever rounds have
     actually happened (with a real W-L series record and winner) and
     leaving future rounds genuinely blank (TBD) rather than guessed.
  2. PROJECTED SEEDING (playoff_bracket) - only used when path 1 finds
     NO real postseason games at all (i.e. the season's still in its
     regular-season phase). Implements the real 2022+ postseason format
     (3 division winners seeded 1-3 by record, 3 wild cards seeded 4-6,
     Wild Card Series in the standard 3-vs-6/4-vs-5 cross pattern, seeds 1
     AND 2 both byes into the Division Series) applied to CURRENT
     standings - a live "if the season ended today" projection, not a
     simulation of the remaining schedule. Deeper rounds are shown
     entirely blank - that would mean guessing games that haven't been
     played, which this project's own convention (see
     api/routers/players.py's wRC+/xFIP notes) avoids: real data or
     nothing, no fabricated results.
Both paths return the SAME shape (see real_postseason_bracket's docstring)
so the API/frontend don't need to know which one answered.

The projected roster is a genuine first attempt at "who would be on this
team's playoff roster right now": lineup/bench/rotation/bullpen slots
filled by real current-season usage (PA for batters, games started/pitched
for pitchers - see analytics/metrics.py's classify_pitcher_roles), for
whichever of the team's players have actually recorded a real pitch this
season and aren't currently on the IL (il_status). This is NOT a simulation
of the real 26-man roster-construction process (bullpen size, catcher
depth, etc. aren't modeled precisely) and is explicitly labeled "Projected"
by the API/frontend rather than presented as official. There is no real
data source in this project's pipeline for the ACTUAL announced postseason
roster once one exists (that's a real-world roster-move fact, not
Statcast-derivable) - see the module's ROSTER note below for how that's
handled instead of being silently faked.
"""
import pandas as pd
import requests

from analytics.boxscore import season_era, season_primary_position
from analytics.metrics import classify_pitcher_roles
from analytics.player_page import batter_counting_stats, pitcher_counting_stats

LINEUP_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
BENCH_SIZE = 4
ROTATION_SIZE = 3   # top 3 only, per direction - not the full 5-man rotation
BULLPEN_SIZE = 7
REGULAR_SEASON_GAMES = 162  # a team logging more than this many real games this season is a live signal real postseason games have started
WC_RACE_DISPLAY_LIMIT = 10  # the race is only meaningful near the cutline - per direction, the list doesn't need to go past 10

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
POSTSEASON_GAME_TYPES = "F,D,L,W"  # F=Wild Card, D=Division Series, L=Championship Series, W=World Series


def _head_to_head_wins(conn, season, team_a, team_b):
    """Real head-to-head season series record between two teams, straight
    from this project's own games table - the actual FIRST tiebreaker real
    MLB uses for a standings tie (NOT run differential, which is what this
    replaces - confirmed a real, not hypothetical, miss: the real 2025
    Blue Jays beat the Yankees for the AL East on a real head-to-head
    tiebreaker, 94-68 apiece, despite the Yankees having a far better run
    differential, +164 vs +77 - the run_diff tiebreak this project used to
    have would have picked the Yankees, which is factually wrong). Returns
    (team_a_wins, team_b_wins)."""
    rows = conn.execute("""
        SELECT home_team, away_team, final_score FROM games
        WHERE season = ? AND (
            (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)
        )
    """, (season, str(team_a), str(team_b), str(team_b), str(team_a))).fetchall()
    a_wins = b_wins = 0
    for home, away, score in rows:
        if not score:
            continue
        try:
            home_score, away_score = (int(x) for x in score.split("-"))
        except ValueError:
            continue
        winner = home if home_score > away_score else away
        if str(winner) == str(team_a):
            a_wins += 1
        elif str(winner) == str(team_b):
            b_wins += 1
    return a_wins, b_wins


def _sorted_ranked(conn, season, div_df):
    """Ranks by win_pct (run_diff as a distant fallback ONLY for breaking a
    tie head-to-head can't resolve), but an exact win_pct tie between two
    adjacent teams is re-checked against their real head-to-head record
    first - real MLB's actual tiebreaker, not run differential (see
    _head_to_head_wins). Only handles a clean pairwise tie precisely; a
    genuine 3+-way tie (rare) falls through to the run_diff ordering below
    it - real MLB's own multi-team tiebreaker math is considerably more
    involved than this first pass implements, flagged here rather than
    silently guessed at."""
    ranked = div_df.sort_values(["win_pct", "run_diff"], ascending=[False, False]).reset_index(drop=True)
    i = 0
    while i < len(ranked) - 1:
        if ranked.loc[i, "win_pct"] == ranked.loc[i + 1, "win_pct"]:
            a_id, b_id = int(ranked.loc[i, "team_id"]), int(ranked.loc[i + 1, "team_id"])
            a_wins, b_wins = _head_to_head_wins(conn, season, a_id, b_id)
            if b_wins > a_wins:
                ranked.loc[[i, i + 1]] = ranked.loc[[i + 1, i]].values
        i += 1
    return ranked


def division_leaders(conn, season, standings_df):
    """One row per division: the current leader plus their lead (in games)
    over the second-place team. Standard games-back formula:
    ((leader_w - leader_l) - (second_w - second_l)) / 2."""
    out = []
    for division, div_df in standings_df.groupby("division"):
        ranked = _sorted_ranked(conn, season, div_df)
        leader = ranked.iloc[0]
        lead = None
        if len(ranked) > 1:
            second = ranked.iloc[1]
            lead = ((leader["wins"] - leader["losses"]) - (second["wins"] - second["losses"])) / 2
        out.append({
            "division": division,
            "league": leader["league"],
            "team_id": int(leader["team_id"]),
            "name": leader["name"],
            "abbreviation": leader["abbreviation"],
            "wins": int(leader["wins"]),
            "losses": int(leader["losses"]),
            "win_pct": round(float(leader["win_pct"]), 3),
            "lead": round(lead, 1) if lead is not None else None,
        })
    return sorted(out, key=lambda d: (d["league"], d["division"]))


def _division_leader_ids(conn, season, league_df):
    ids = set()
    for division, div_df in league_df.groupby("division"):
        ids.add(int(_sorted_ranked(conn, season, div_df).iloc[0]["team_id"]))
    return ids


def wildcard_race(conn, season, standings_df, league):
    """Every non-division-leader team in `league`, ranked by record - real
    MLB rules mean a division leader can never also hold a wild card spot
    even if their record would otherwise qualify, so they're excluded here
    (see division_leaders for their own listing). `in_wc_spot` marks the
    current top 3 - the actual bracket-qualifying cutoff. `games_back` is
    measured from the LAST wild card spot (seed 6, i.e. rank 3 of this
    list) - the real "how close to being knocked out/how close to getting
    in" number; teams already in a wild card spot get games_back=None
    (displayed as "-"), matching the standard convention that a team ahead
    of the cutline doesn't have a deficit to report."""
    league_df = standings_df[standings_df["league"] == league]
    leader_ids = _division_leader_ids(conn, season, league_df)
    contenders = league_df[~league_df["team_id"].astype(int).isin(leader_ids)]
    ranked = _sorted_ranked(conn, season, contenders).reset_index(drop=True)
    if ranked.empty:
        return []
    cutline = ranked.iloc[min(2, len(ranked) - 1)]  # last (3rd) wild card spot
    out = []
    for i, row in ranked.iterrows():
        if i >= WC_RACE_DISPLAY_LIMIT:  # the race is only meaningful near the cutline - no need to list every non-contender down to last place
            break
        gb = ((cutline["wins"] - cutline["losses"]) - (row["wins"] - row["losses"])) / 2
        out.append({
            "team_id": int(row["team_id"]),
            "name": row["name"],
            "abbreviation": row["abbreviation"],
            "division": row["division"],
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "win_pct": round(float(row["win_pct"]), 3),
            "run_diff": int(row["run_diff"]),
            "rank": i + 1,
            "in_wc_spot": i < 3,
            "games_back": None if i < 3 else round(gb, 1),
        })
    return out


def playoff_bracket(conn, season, standings_df, league):
    """Seeds 1-6 for `league` under the real 2022+ format: 3 division
    winners seeded 1-3 by record, 3 wild cards seeded 4-6 by record among
    the rest. Wild Card Series matchups are the standard cross pattern (3
    vs 6, 4 vs 5); seed 1 is a bye straight to the Division Series. Live
    off current standings - "if the season started today," not a
    projection of who WINS anything."""
    league_df = standings_df[standings_df["league"] == league]
    leader_rows = [_sorted_ranked(conn, season, div_df).iloc[0] for _, div_df in league_df.groupby("division")]
    leaders_df = pd.DataFrame(leader_rows)
    leaders_ranked = _sorted_ranked(conn, season, leaders_df)
    leader_ids = set(int(x) for x in leaders_ranked["team_id"])

    wc_pool = league_df[~league_df["team_id"].astype(int).isin(leader_ids)]
    wc_ranked = _sorted_ranked(conn, season, wc_pool).head(3)

    def seed_row(row, seed):
        return {
            "seed": seed, "team_id": int(row["team_id"]), "name": row["name"],
            "abbreviation": row["abbreviation"], "wins": int(row["wins"]),
            "losses": int(row["losses"]), "win_pct": round(float(row["win_pct"]), 3),
            "division": row["division"],
        }

    seeds = [seed_row(row, i + 1) for i, (_, row) in enumerate(leaders_ranked.iterrows())]
    seeds += [seed_row(row, i + 4) for i, (_, row) in enumerate(wc_ranked.iterrows())]
    by_seed = {s["seed"]: s for s in seeds}

    matchups = []
    if 3 in by_seed and 6 in by_seed:
        matchups.append({"higher": by_seed[3], "lower": by_seed[6]})
    if 4 in by_seed and 5 in by_seed:
        matchups.append({"higher": by_seed[4], "lower": by_seed[5]})

    return {"league": league, "bye": by_seed.get(1), "wild_card_matchups": matchups, "seeds": seeds}


def playoff_teams(conn, season, standings_df, league):
    """The 6 team_ids currently in bracket position for `league` - used by
    the roster tab's team picker so it only offers teams actually in the
    projected field."""
    bracket = playoff_bracket(conn, season, standings_df, league)
    return [s["team_id"] for s in bracket["seeds"]]


def postseason_started(standings_df):
    """Live signal that real postseason games exist in this project's own
    data (any team's real games-played count exceeding a normal 162-game
    season) - drives the projected-roster label per-team. The Bracket
    page's own title uses real_postseason_bracket's result directly
    instead (a more direct signal - see that function)."""
    if standings_df.empty:
        return False
    return bool((standings_df["games_played"] > REGULAR_SEASON_GAMES).any())


# ------------------------------------------------------------- Real bracket

def _fetch_postseason_games(season):
    """Every completed ("Final") postseason game for `season`, straight
    from the MLB Stats API's own schedule endpoint - same official,
    unauthenticated live-API pattern analytics/boxscore.py's season_era
    already uses. Returns [] (not an error) for a season with no real
    postseason games yet (current in-progress season) or a network hiccup -
    the caller falls back to the projected bracket either way."""
    try:
        resp = requests.get(
            SCHEDULE_URL, params={"sportId": 1, "season": season, "gameType": POSTSEASON_GAME_TYPES}, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    games = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            desc = g.get("seriesDescription") or ""
            home, away = g["teams"]["home"], g["teams"]["away"]
            games.append({
                "series_desc": desc,
                "games_in_series": g.get("gamesInSeries"),
                "home_id": home["team"]["id"], "home_name": home["team"]["name"], "home_score": home.get("score"),
                "away_id": away["team"]["id"], "away_name": away["team"]["name"], "away_score": away.get("score"),
            })
    return games


def _round_key(desc):
    """Maps a real seriesDescription string to (league, round) - matched by
    substring, not an exact lookup table, so it transparently handles both
    the 2022+ phrasing ("AL Wild Card Series") and the pre-2022 one-game
    play-in phrasing ("AL Wild Card Game") without needing a per-era
    special case - confirmed directly against real 2021 and 2022 schedule
    data before relying on this."""
    league = "al" if desc.startswith("AL") else "nl" if desc.startswith("NL") else None
    if "Wild Card" in desc:
        return league, "wildcard"
    if "Division Series" in desc:
        return league, "division"
    if "Championship Series" in desc:
        return league, "championship"
    if desc == "World Series":
        return None, "world_series"
    return None, None


def _matchup_from_games(games):
    """One series' real result from its own completed games: both real
    team names/ids, the real win-loss record so far, and the real winner
    ONLY once one side has actually clinched (win count reaches the real
    gamesInSeries majority) - a 2-1 series in progress correctly reports no
    winner yet rather than guessing from a lead."""
    team_names, wins, games_in_series = {}, {}, None
    for g in games:
        team_names[g["home_id"]] = g["home_name"]
        team_names[g["away_id"]] = g["away_name"]
        if g.get("games_in_series"):
            games_in_series = g["games_in_series"]
        if g["home_score"] is None or g["away_score"] is None:
            continue
        winner_id = g["home_id"] if g["home_score"] > g["away_score"] else g["away_id"]
        wins[winner_id] = wins.get(winner_id, 0) + 1
    ids = list(team_names.keys())
    if len(ids) != 2:
        return None
    a_id, b_id = ids
    win_a, win_b = wins.get(a_id, 0), wins.get(b_id, 0)
    clinch = games_in_series // 2 + 1 if games_in_series else None
    winner_id = a_id if clinch and win_a >= clinch else b_id if clinch and win_b >= clinch else None
    return {
        "team_a": {"team_id": a_id, "name": team_names[a_id]},
        "team_b": {"team_id": b_id, "name": team_names[b_id]},
        "winner_team_id": winner_id,
        "record": f"{win_a}-{win_b}" if (win_a or win_b) else None,
    }


def real_postseason_bracket(season):
    """Real bracket + results for `season`, straight from completed games -
    or None if no real postseason games exist yet for this season (the
    caller falls back to the projected bracket). Returns:
        {"al": {"wildcard": [...], "division": [...], "championship": [...]},
         "nl": {same shape},
         "world_series": {...} or None}
    Each matchup dict is {team_a, team_b, winner_team_id, record} - see
    _matchup_from_games. Rounds that haven't happened yet simply have an
    empty/short list (or world_series=None) rather than a guessed entry."""
    games = _fetch_postseason_games(season)
    if not games:
        return None

    buckets = {}
    for g in games:
        league, kind = _round_key(g["series_desc"])
        if kind is None:
            continue
        key = (league, kind, frozenset({g["home_id"], g["away_id"]}))
        buckets.setdefault(key, []).append(g)

    out = {"al": {"wildcard": [], "division": [], "championship": []},
           "nl": {"wildcard": [], "division": [], "championship": []},
           "world_series": None}
    for (league, kind, _), gs in buckets.items():
        matchup = _matchup_from_games(gs)
        if matchup is None:
            continue
        if kind == "world_series":
            out["world_series"] = matchup
        else:
            out[league][kind].append(matchup)

    # wildcard[i] and division[i] arrive in whatever order Python's dict
    # iteration happened to produce (insertion order off the schedule
    # feed's own date ordering) - NOT necessarily "these two rounds are
    # the same bracket line." The frontend draws a straight connector from
    # wildcard[i] to division[i], so a real mismatch here draws a real
    # wrong line (confirmed bug: 2025 AL had wildcard[1] - Yankees/Red Sox
    # - visually paired with division[1] - Mariners/Tigers - when the real
    # winner, the Yankees, actually advanced into division[0] against the
    # Blue Jays). Fixed by re-sorting wildcard so wildcard[i]'s WINNER is
    # one of division[i]'s two participants - the one fact that's always
    # true by definition once both rounds are decided. Only reorders when
    # it's actually the standard 2-vs-2 shape with a real winner to check
    # against; an in-progress Wild Card round (winner still unknown) has
    # no way to know this yet either, in reality, so it's left alone.
    for league in ("al", "nl"):
        wc, ds = out[league]["wildcard"], out[league]["division"]
        if len(wc) == 2 and len(ds) == 2 and wc[0]["winner_team_id"] is not None:
            ds_ids = [{t["team_id"] for t in (m["team_a"], m["team_b"]) if t} for m in ds]
            if wc[0]["winner_team_id"] not in ds_ids[0] and wc[0]["winner_team_id"] in ds_ids[1]:
                wc[0], wc[1] = wc[1], wc[0]

    # World Series team_a/team_b arrive in whatever order the schedule feed
    # happened to list home/away in, NOT reliably NL-then-AL - the frontend
    # needs a consistent side (NL always drawn on the left, AL on the
    # right), so pin team_a to whichever side's own championship-round
    # participants it matches.
    ws = out["world_series"]
    if ws and ws["team_a"] is not None:
        nl_ids = {t["team_id"] for m in out["nl"]["championship"] for t in (m["team_a"], m["team_b"]) if t}
        if ws["team_a"]["team_id"] not in nl_ids and ws["team_b"] and ws["team_b"]["team_id"] in nl_ids:
            ws["team_a"], ws["team_b"] = ws["team_b"], ws["team_a"]

    return out


def _team_ref(seed_row):
    return None if seed_row is None else {"team_id": seed_row["team_id"], "name": seed_row["name"]}


def _empty_matchup():
    return {"team_a": None, "team_b": None, "winner_team_id": None, "record": None}


def _projected_unified_bracket(conn, season, standings_df):
    """The standings-projected bracket (playoff_bracket), reshaped into the
    same {al: {wildcard, division, championship}, nl: {...}, world_series}
    shape real_postseason_bracket returns, so the API layer can hand back
    one consistent contract regardless of which path answered. Seeds 1 and
    2 both appear in "division" (both are real byes under the current
    format - see playoff_bracket's own docstring); their Division Series
    opponent is genuinely unknown so team_b stays None (TBD)."""
    def convert(league):
        b = playoff_bracket(conn, season, standings_df, league)
        by_seed = {s["seed"]: s for s in b["seeds"]}
        wildcard = [
            {"team_a": _team_ref(m["higher"]), "team_b": _team_ref(m["lower"]), "winner_team_id": None, "record": None}
            for m in b["wild_card_matchups"]
        ]
        division = [
            {"team_a": _team_ref(by_seed.get(1)), "team_b": None, "winner_team_id": None, "record": None},
            {"team_a": _team_ref(by_seed.get(2)), "team_b": None, "winner_team_id": None, "record": None},
        ]
        return {"wildcard": wildcard, "division": division, "championship": [_empty_matchup()]}

    return {"al": convert("AL"), "nl": convert("NL"), "world_series": _empty_matchup()}


def unified_bracket(conn, standings_df, season):
    """The single entry point the API layer calls: real results when any
    exist for `season`, else the live standings projection. See this
    module's docstring for why these are two genuinely different things,
    not one path pretending to cover both."""
    real = real_postseason_bracket(season)
    if real is not None:
        return {"season": season, "is_real": True, "postseason_started": True, **real}
    projected = _projected_unified_bracket(conn, season, standings_df)
    return {"season": season, "is_real": False, "postseason_started": postseason_started(standings_df), **projected}


# --------------------------------------------------------- Projected roster

def _current_season(conn):
    row = conn.execute("SELECT MAX(season) FROM pitches").fetchone()
    return row[0] if row else None


def _il_ids(conn, season):
    """IL status is a REAL-TIME fact (who's hurt RIGHT NOW), not a
    per-season historical one - only meaningful for the live season still
    being played. Applying it to a past completed season wrongly dropped a
    healthy-then/hurt-now player from that real season's roster - a real
    bug (Aaron Judge missing from the real 2025 Yankees roster because
    he's on the IL as of today, 2026) confirmed and fixed, not a
    hypothetical. Only the current season's roster is filtered by it."""
    if season != _current_season(conn):
        return set()
    return {r[0] for r in conn.execute("SELECT player_id FROM il_status").fetchall()}


def _player_season_teams(conn, season, id_col, side_expr):
    """{player_id: team_id} - each player's OWN team for THIS season
    specifically, derived from real per-game home/away context (games.
    home_team/away_team + pitches.inning_topbot: 'Top' = home team
    pitching/away team batting, 'Bot' = the reverse), not players.team_id
    (which only ever reflects a player's CURRENT/most-recent team - wrong
    for a past season, and wrong for anyone traded mid-season even in the
    current one). A player who played for two teams this season is
    attributed to whichever he played MORE for (most real pitches/PAs),
    the same real-usage-wins convention this whole roster feature already
    uses elsewhere. Confirmed directly against a known real player
    (Gerrit Cole -> 147/Yankees, single-team season) before relying on
    this, not assumed. `side_expr` is the SQL CASE picking the right side
    for batter vs pitcher (see the two call sites below)."""
    rows = conn.execute(f"""
        SELECT pt.{id_col} AS player_id, {side_expr} AS team_id, COUNT(*) AS n
        FROM pitches pt JOIN games g ON g.game_id = pt.game_id
        WHERE pt.season = ?
        GROUP BY pt.{id_col}, team_id
    """, (season,)).fetchall()
    best = {}
    for pid, team_id, n in rows:
        if pid not in best or n > best[pid][1]:
            best[pid] = (team_id, n)
    return {pid: team for pid, (team, _) in best.items()}


def _team_affiliation(conn, season, id_col, side_expr):
    """Which team each player counts as part of, for roster purposes, in
    `season` - deliberately NOT the same answer for every season:
      - CURRENT/live season: players.team_id (who's really on the roster
        RIGHT NOW) - "who'd make the playoff roster" is a question about
        today's roster, so a player traded in mid-stretch-run belongs to
        his NEW team even if most of this season's stats happened
        elsewhere. Confirmed as a real, not hypothetical, case: Liam
        Hicks, traded to Tampa Bay in August, had 1496 of his 1758
        pitches this season as a Marlin but is genuinely a Ray now -
        _player_season_teams' PA-majority rule was putting him on the
        wrong team for this page's actual purpose.
      - PAST/completed season: players.team_id only ever reflects
        whoever a player is on TODAY, which is simply wrong for
        history (see _player_season_teams' own docstring) - use the
        real per-game-derived season team instead. Confirmed as a real
        bug this replaces: Aaron Judge was missing from the real 2025
        Yankees roster because of this exact mismatch.
    """
    if season == _current_season(conn):
        return {pid: team for pid, team in conn.execute("SELECT player_id, team_id FROM players").fetchall()}
    return _player_season_teams(conn, season, id_col, side_expr)


def _season_position(conn, season, player_id, fallback_position):
    """players.position is only ever the player's CURRENT real-world
    position (overwritten on every roster pull - see
    pulls/pull_players_games.py), so it's only trustworthy for the live
    current season. For a past season it can be flatly wrong for anyone
    who's since changed position (e.g. Bo Bichette: players.position says
    '3B', his current Mets spot, but he was really the Blue Jays' SS in
    2025) - which then wrongly collides that player with a teammate who
    really did hold that position, bumping the real starter to the bench.
    Live per-season fielding split, same "most usage wins" logic
    _team_affiliation already uses for team; a network hiccup just falls
    back to the static field rather than breaking the whole roster."""
    if season == _current_season(conn):
        return fallback_position
    try:
        pos = season_primary_position(player_id, season)
    except Exception:
        pos = None
    return pos or fallback_position


def _team_batters(conn, season, team_id):
    """Every player whose team this season (see _team_affiliation) is
    `team_id` and who's recorded a real plate appearance - the real basis
    for "who's actually played here," correct for past seasons and
    mid-season trades alike."""
    il = _il_ids(conn, season)
    team_by_player = _team_affiliation(conn, season, "batter_id", "CASE WHEN pt.inning_topbot = 'Top' THEN g.away_team ELSE g.home_team END")
    rows = conn.execute("""
        SELECT p.player_id, p.name, p.position, COUNT(*) AS n_pa
        FROM players p
        JOIN pitches pt ON pt.batter_id = p.player_id AND pt.season = ? AND pt.events IS NOT NULL
        GROUP BY p.player_id, p.name, p.position
    """, (season,)).fetchall()
    return [
        dict(player_id=r[0], name=r[1], position=_season_position(conn, season, r[0], r[2]), n_pa=r[3])
        for r in rows if r[0] not in il and str(team_by_player.get(r[0])) == str(team_id)
    ]


def _team_pitchers(conn, season, team_id, roles_df):
    il = _il_ids(conn, season)
    team_by_player = _team_affiliation(conn, season, "pitcher_id", "CASE WHEN pt.inning_topbot = 'Top' THEN g.home_team ELSE g.away_team END")
    rows = conn.execute("""
        SELECT DISTINCT p.player_id, p.name
        FROM players p
        JOIN pitches pt ON pt.pitcher_id = p.player_id AND pt.season = ?
    """, (season,)).fetchall()
    out = []
    for pid, name in rows:
        if pid in il or pid not in roles_df.index or str(team_by_player.get(pid)) != str(team_id):
            continue
        r = roles_df.loc[pid]
        out.append(dict(
            player_id=pid, name=name, role=r["role"],
            games_started=int(r["games_started"]), games_pitched=int(r["games_pitched"]),
        ))
    return out


def _build_lineup(batters):
    by_pos, used = {}, set()
    for pos in LINEUP_POSITIONS:
        candidates = [b for b in batters if b["position"] == pos]
        best = max(candidates, key=lambda b: b["n_pa"]) if candidates else None
        by_pos[pos] = best
        if best:
            used.add(best["player_id"])
    bench_pool = sorted((b for b in batters if b["player_id"] not in used), key=lambda b: -b["n_pa"])
    return by_pos, bench_pool


def _war_lookup(conn, season):
    rows = conn.execute(
        "SELECT player_id, war_batting, war_pitching FROM player_advanced_stats WHERE season = ?",
        (season,),
    ).fetchall()
    return {pid: (wb, wp) for pid, wb, wp in rows}


def _batter_stat_line(conn, season, player_id, name, position, war_lookup):
    stats = batter_counting_stats(conn, season, player_id)
    war = war_lookup.get(player_id, (None, None))[0]
    return {
        "player_id": player_id, "name": name, "position": position,
        "pa": stats["n_pa"], "hr": stats["hr"],
        "ops": None if stats["OPS"] is None or pd.isna(stats["OPS"]) else round(float(stats["OPS"]), 3),
        "woba": None if stats["wOBA"] is None or pd.isna(stats["wOBA"]) else round(float(stats["wOBA"]), 3),
        "war": None if war is None or pd.isna(war) else round(float(war), 1),
    }


def _pitcher_stat_line(conn, season, player_id, name, role_row):
    stats = pitcher_counting_stats(conn, season, player_id, games_started=role_row["games_started"])
    n_pa = stats["n_pa"]
    era = None
    try:
        era = season_era(player_id, season)
    except Exception:
        era = None  # live external call - a network hiccup shouldn't break the whole roster
    return {
        "player_id": player_id, "name": name,
        "ip": stats["ip"],
        "games_started": role_row["games_started"],
        "games_pitched": role_row["games_pitched"],
        "era": None if era is None else round(float(era), 2),
        "k_pct": None if not n_pa else round(stats["k"] / n_pa * 100, 1),
        "bb_pct": None if not n_pa else round(stats["bb"] / n_pa * 100, 1),
        "war": None,  # filled in by the caller (batch war_lookup, same as batters)
    }


def projected_roster(conn, season, team_id, games_played_this_team=None):
    """Lineup (by position, highest-PA player at each) / bench / rotation
    (top 3 by games started) / bullpen (top BULLPEN_SIZE by games
    pitched) for `team_id`, built from real current-season usage among
    players who've actually recorded a pitch for this team this season and
    aren't on the IL. `games_played_this_team`: if the caller already has
    this team's games-played count (e.g. from standings), pass it so the
    "has the postseason started" label below doesn't need a second query.

    ROSTER-SWITCH NOTE: this project has no real data source for a team's
    ACTUAL announced postseason active roster (a real front-office
    decision, not something Statcast publishes) - so rather than fabricate
    one, `is_real_postseason_roster` below is just a label: once this
    team's real games-played count exceeds a normal 162-game season (the
    live, honest signal that postseason games are actually being played),
    the roster still reflects the same real-usage projection, just
    relabeled from "Projected" to reflect that real playoff games are
    underway. See this module's own docstring."""
    roles_df = classify_pitcher_roles(conn, season)
    batters = _team_batters(conn, season, team_id)
    pitchers = _team_pitchers(conn, season, team_id, roles_df)
    war_lookup = _war_lookup(conn, season)

    by_pos, bench_pool = _build_lineup(batters)
    bench = bench_pool[:BENCH_SIZE]

    starters = sorted((p for p in pitchers if p["role"] == "SP"), key=lambda p: -p["games_started"])[:ROTATION_SIZE]
    relievers = sorted((p for p in pitchers if p["role"] == "RP"), key=lambda p: -p["games_pitched"])[:BULLPEN_SIZE]

    lineup = []
    for pos in LINEUP_POSITIONS:
        b = by_pos[pos]
        lineup.append(_batter_stat_line(conn, season, b["player_id"], b["name"], pos, war_lookup) if b else {"position": pos, "player_id": None})

    bench_out = [_batter_stat_line(conn, season, b["player_id"], b["name"], b["position"], war_lookup) for b in bench]

    rotation_out = []
    for p in starters:
        line = _pitcher_stat_line(conn, season, p["player_id"], p["name"], p)
        line["war"] = war_lookup.get(p["player_id"], (None, None))[1]
        if line["war"] is not None and not pd.isna(line["war"]):
            line["war"] = round(float(line["war"]), 1)
        else:
            line["war"] = None
        rotation_out.append(line)

    bullpen_out = []
    for p in relievers:
        line = _pitcher_stat_line(conn, season, p["player_id"], p["name"], p)
        line["war"] = war_lookup.get(p["player_id"], (None, None))[1]
        if line["war"] is not None and not pd.isna(line["war"]):
            line["war"] = round(float(line["war"]), 1)
        else:
            line["war"] = None
        bullpen_out.append(line)

    is_postseason = bool(games_played_this_team and games_played_this_team > REGULAR_SEASON_GAMES)

    return {
        "team_id": team_id,
        "season": season,
        "is_real_postseason_roster": is_postseason,
        "lineup": lineup,
        "bench": bench_out,
        "rotation": rotation_out,
        "bullpen": bullpen_out,
    }
