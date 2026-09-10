"""
Team card popup - click any team logo anywhere on the site (wired via
frontend/js/team-card.js, event-delegated so it works without every page
needing its own click handler) to see the org's general approach plus its
season rank (of 30) and trailing-30-day snapshot in OPS/ERA/bullpen ERA/FIP.

Ranks are computed live, on demand, from team_stats_snapshot (pulled by
pull_team_stats.py - see that file's docstring for why it's MLB Stats API
sourced rather than FanGraphs, which turned out to be unreachable). Not
cached at the API layer the way the big per-request pools are - a 30-row
table scan is cheap enough not to need it.

TEAM_SYNOPSES is hand-written, one to two sentences per team, based on each
org's real, broadly-reported public strategic identity (payroll approach,
player-development philosophy, roster-construction pattern) - not derived
from a stat, and not claiming to be exhaustive or authoritative. Treat it
the same way as this site's other descriptive-not-computed content (e.g.
the archetype descriptions).
"""
import numpy as np
import pandas as pd

from analytics.strength_of_schedule import remaining_sos

TEAM_SYNOPSES = {
    "AZ": "Aggressive, opportunistic roster-building around a homegrown core - willing to trade prospect "
          "depth for a win-now piece when a window looks real, as the 2023 pennant run showed.",
    "ATH": "Full rebuild in a new home market (Sacramento, en route to Las Vegas) - bottom-of-the-league "
          "payroll, developmental patience, and public focus on the eventual permanent ballpark over "
          "short-term results.",
    "ATL": "Player-development engine that locks up its own young core to team-friendly long-term extensions "
          "early, rather than letting them reach free agency - a stable, continuity-first model built around "
          "a strong farm system.",
    "BAL": "Analytics-forward rebuild that leaned hard on the draft and international amateur scouting to "
          "restock the system, then graduated a wave of position-player prospects together rather than "
          "spreading debuts out.",
    "BOS": "Big-market payroll with a self-imposed taste for financial discipline (repeated luxury-tax-line "
          "targeting) that creates real tension with the fan base's championship expectations.",
    "CHC": "Historic big-market franchise cycling between aggressive free-agent spending and leaner "
          "development-first stretches, with Wrigley Field's identity as a constant regardless of the "
          "roster's competitive window.",
    "CWS": "Rebuilding from a low base after a rapid competitive collapse - accumulating young pitching and "
          "positional depth through the draft and trades, with patience the organizing principle for now.",
    "CIN": "Small-market club leaning on a homegrown, athletic young core and an aggressive, fast-paced style "
          "of play, extending its own core rather than competing in free agency for outside stars.",
    "CLE": "Perennial pitching-development and reclamation shop - takes on modest-profile arms and remakes "
          "their mechanics/pitch mix, while running one of the league's lowest payrolls by design.",
    "COL": "Altitude-shaped roster-building (offense inflated, pitching suppressed by Coors Field) that makes "
          "typical park-neutral stats read differently for this org than any other - development has "
          "struggled to translate playing half a season at elevation.",
    "DET": "Pitching-development-first rebuild that produced a wave of front-line young arms, now pairing "
          "that with opportunistic aggression in trades and free agency as the competitive window opens.",
    "HOU": "Analytics pioneer with an heavy emphasis on pitch design and swing optimization - sustained "
          "success built on player-development infrastructure as much as any single star, through real "
          "roster turnover.",
    "KC": "Small-market club built around contact-oriented, athletic position players and a defense-first "
          "philosophy up the middle, pairing homegrown pitching with occasional trades for a proven arm.",
    "LAA": "Star-driven roster construction (built for years around two of the sport's best players) that has "
          "repeatedly struggled to develop a supporting young core deep enough to complement them.",
    "LAD": "Baseball's deepest payroll combined with genuinely elite player development - out-spends the "
          "league AND out-develops it, giving them roster flexibility most orgs can't match.",
    "MIA": "Low-payroll, prospect-accumulation model - graduates young pitching and outfield talent, then "
          "frequently trades established players at peak value to restock the pipeline before they hit "
          "free agency.",
    "MIL": "Model small-market club - wins with pitching depth, shrewd bullpen usage, and aggressive, "
          "metrics-driven in-game strategy (shifts, openers) rather than payroll, extending stars only "
          "selectively.",
    "MIN": "Mid-market club balancing a homegrown position-player core with occasional star free-agent "
          "additions, generally competing for the AL Central rather than outspending the league's top payrolls.",
    "NYM": "High-payroll, ownership-driven aggression in free agency paired with a retooled, analytics-heavy "
          "front office investing seriously in player development infrastructure for the first time in years.",
    "NYY": "Historic large-market power/pitching identity - built around thump in the middle of the order and "
          "a willingness to spend at the top of the market, while player development fills in around "
          "free-agent anchors.",
    "PHI": "Aggressive, win-now free-agent spending on a veteran core built to contend immediately, with less "
          "organizational patience for a rebuilding cycle than most.",
    "PIT": "Perennial small-payroll rebuild leaning on the draft for front-line pitching talent, with a real "
          "tension between fan patience and ownership's spending level.",
    "SD": "High-payroll aggression for a mid-size market - repeatedly all-in on the current roster via trades "
          "and free agency, prioritizing a near-term contention window over long-term payroll flexibility.",
    "SF": "Analytically-driven, swing-and-approach-optimization player development, historically favoring "
          "a deep, versatile roster over a small number of true superstars.",
    "SEA": "Pitching-and-defense-oriented identity shaped by a spacious home park, with a well-regarded "
          "developmental pipeline that has had real trouble translating into sustained division contention.",
    "STL": "Traditionally continuity-first, homegrown-pitching-and-defense organization now in a more overt "
          "retooling phase after a rare sustained down stretch broke from its usual competitive norm.",
    "TB": "The sport's defining low-payroll, high-efficiency model - extreme roster flexibility, heavy "
          "platooning, aggressive bullpen usage (including openers), and a track record of competing on a "
          "fraction of the league's median payroll.",
    "TEX": "Recent aggressive, ownership-backed free-agent spending (culminating in a 2023 title) paired with "
          "a still-developing young pitching pipeline meant to sustain the window that spending opened.",
    "TOR": "Mid-to-large payroll built around a homegrown offensive core supplemented by free-agent additions, "
          "playing under the unique cross-border/travel logistics of being MLB's only non-U.S. franchise.",
    "WSH": "Rebuilding from a low base after its championship core was traded away - leaning on prospect "
          "capital acquired in those deals while payroll stays near the bottom of the league.",
}


def _rank(rows, key, ascending):
    """1-indexed rank of every team on `key` (lower=1 if ascending, i.e. ERA/
    FIP where lower is better; higher=1 if not, i.e. OPS where higher is
    better). Returns {team_id: rank}. Rows missing this key are excluded
    from ranking entirely (not sorted to one end) rather than crashing on a
    None comparison."""
    ranked_rows = [r for r in rows if r.get(key) is not None]
    ordered = sorted(ranked_rows, key=lambda r: r[key], reverse=not ascending)
    return {r["team_id"]: i + 1 for i, r in enumerate(ordered)}


def _team_war(conn, season):
    """Team total WAR (batting + pitching summed per player, then per team) -
    player_advanced_stats.war_batting/war_pitching are real MLB Stats API
    sabermetrics totals (see schema.sql), not derived here. players.team_id
    is stored as TEXT (existing schema quirk) so it's cast to match
    team_stats_snapshot/teams' INTEGER team_id for the lookup dict below."""
    rows = conn.execute("""
        SELECT p.team_id, SUM(COALESCE(a.war_batting, 0) + COALESCE(a.war_pitching, 0))
        FROM player_advanced_stats a JOIN players p ON p.player_id = a.player_id
        WHERE a.season = ? AND p.team_id IS NOT NULL
        GROUP BY p.team_id
    """, (season,)).fetchall()
    return {int(tid): war for tid, war in rows if tid is not None and str(tid).isdigit()}


def _team_il_counts(conn):
    """Current IL headcount per team - see pulls/pull_injuries.py."""
    rows = conn.execute("SELECT team_id, COUNT(*) FROM il_status GROUP BY team_id").fetchall()
    return {tid: n for tid, n in rows}


PARK_FACTOR_MIN_GAMES = 10

def _team_park_factors(conn, season):
    """Real, SELF-COMPUTED single-season "basic" runs-based park factor -
    100 * (runs scored+allowed per HOME game) / (runs scored+allowed per
    ROAD game), using this project's own real, already-ingested game
    results (games.final_score, "home-away" per analytics/standings.py's
    and analytics/wpa.py's own established parsing convention). 100 =
    neutral; above 100 = more runs at home than on the road (hitter-
    friendly); below = pitcher-friendly.

    Explicitly the SIMPLE single-team, single-season method, not the
    multi-year, opponent-adjusted, regressed version real park-factor
    sites publish - real numbers from real games, not fabricated, but a
    genuine approximation, same honest-caveat treatment this project
    already gives xFIP/wRC+ (see api/routers/players.py). No reliable
    ready-made park factor exists from either data source this project
    already reaches (MLB Stats API has no park-factor stat type at all;
    Baseball Savant publishes one but only via client-side JS, not a
    plain HTTP GET - confirmed directly, not assumed). Computed here
    directly from the already-ingested `games` table rather than a new
    pull script, since nothing external needs fetching.
    """
    df = pd.read_sql(
        "SELECT home_team, away_team, final_score FROM games WHERE season = ? AND final_score IS NOT NULL",
        conn, params=(season,),
    )
    if df.empty:
        return {}
    df[["home_score", "away_score"]] = df["final_score"].str.split("-", expand=True).astype(int)

    home = df.groupby("home_team").agg(
        home_games=("home_score", "size"), home_runs=("home_score", "sum"), home_runs_allowed=("away_score", "sum"),
    )
    away = df.groupby("away_team").agg(
        away_games=("away_score", "size"), away_runs=("away_score", "sum"), away_runs_allowed=("home_score", "sum"),
    )
    combined = home.join(away, how="outer").fillna(0)
    home_rpg = (combined["home_runs"] + combined["home_runs_allowed"]) / combined["home_games"].replace(0, np.nan)
    away_rpg = (combined["away_runs"] + combined["away_runs_allowed"]) / combined["away_games"].replace(0, np.nan)
    pf = (home_rpg / away_rpg * 100)
    enough_games = (combined["home_games"] >= PARK_FACTOR_MIN_GAMES) & (combined["away_games"] >= PARK_FACTOR_MIN_GAMES)
    pf = pf.where(enough_games).round(1)
    return {int(tid): v for tid, v in pf.dropna().items() if str(tid).isdigit()}


# Every ranked metric this card supports, and how to rank it (ascending =
# lower raw value is rank 1). Shared by team_card() (one team's own numbers)
# and team_metric_rankings() (the full 30-team list behind the per-metric
# expand arrow) so the two never define a metric's ranking direction
# differently.
RANKED_METRICS = {
    "ops": False, "era": True, "bullpen_era": True, "fip": True,
    "baserunning_value": False,
    "war": False,          # 1st = most team WAR
    "il_count": True,      # 1st = fewest players on the IL
    "park_factor": False,  # 1st = most hitter-friendly
}


def _all_team_rows_and_ranks(conn, season):
    """Every team's raw metric values plus the resulting 1-30 ranks - the
    shared computation behind both team_card() (one team's numbers) and
    team_metric_rankings() (the full league list for one metric)."""
    rows = [
        dict(team_id=r[0], name=r[1], abbreviation=r[2], ops=r[3], era=r[4], bullpen_era=r[5], fip=r[6],
             ops_30d=r[7], era_30d=r[8], fip_30d=r[9], baserunning_value=r[10])
        for r in conn.execute("""
            SELECT t.team_id, t.name, t.abbreviation, s.ops, s.era, s.bullpen_era, s.fip,
                   s.ops_30d, s.era_30d, s.fip_30d, s.baserunning_value
            FROM team_stats_snapshot s JOIN teams t ON t.team_id = s.team_id
            WHERE s.season = ?
        """, (season,)).fetchall()
    ]
    if not rows:
        return None, None
    war_by_team = _team_war(conn, season)
    il_by_team = _team_il_counts(conn)
    park_factor_by_team = _team_park_factors(conn, season)
    for r in rows:
        r["war"] = war_by_team.get(r["team_id"])
        r["il_count"] = il_by_team.get(r["team_id"])
        r["park_factor"] = park_factor_by_team.get(r["team_id"])
    ranks = {key: _rank(rows, key, ascending) for key, ascending in RANKED_METRICS.items()}
    return rows, ranks


def team_metric_rankings(conn, season, metric, rows_and_ranks=None):
    """Full 30-team ranked list for one metric - powers the team card's
    per-metric "expand to see everyone" dropdown. Teams missing this metric
    are excluded (same as _rank() itself). `rows_and_ranks`: optional
    pre-fetched (rows, ranks) tuple from _all_team_rows_and_ranks - the
    caller (api/routers/team_card.py) fetches it through its own
    @ttl_cache()'d wrapper rather than every call re-running the season's
    WAR/IL/park-factor computation from scratch; omit to compute it fresh
    here as before."""
    if metric not in RANKED_METRICS:
        return []
    rows, ranks = rows_and_ranks if rows_and_ranks is not None else _all_team_rows_and_ranks(conn, season)
    if rows is None:
        return []
    rank_map = ranks[metric]
    out = [
        {"team_id": r["team_id"], "name": r["name"], "abbreviation": r["abbreviation"],
         "value": r[metric], "rank": rank_map[r["team_id"]]}
        for r in rows if r["team_id"] in rank_map
    ]
    out.sort(key=lambda x: x["rank"])
    return out


def team_card(conn, team_id, season, rows_and_ranks=None, sos_by_team=None):
    """`rows_and_ranks`/`sos_by_team`: optional pre-fetched results (see
    team_metric_rankings' own docstring for why) - the router fetches both
    through its own @ttl_cache()'d wrappers; omit either to compute it
    fresh here as before."""
    rows, ranks = rows_and_ranks if rows_and_ranks is not None else _all_team_rows_and_ranks(conn, season)
    if rows is None:
        return None
    n_teams = len(rows)
    team = next((r for r in rows if r["team_id"] == team_id), None)
    if team is None:
        return None

    def metric(key, rolling_key=None):
        return dict(
            value=team[key], rank=ranks[key].get(team_id), n_teams=n_teams,
            rolling_30d=team.get(rolling_key) if rolling_key else None,
        )

    sos_by_team = sos_by_team if sos_by_team is not None else remaining_sos(conn, season)
    sos = sos_by_team.get(team_id)
    sos_metric = dict(
        value=round(sos["avg_opponent_win_pct"], 3) if sos else None,
        rank=sos["rank"] if sos else None,
        n_teams=n_teams,
        n_games_remaining=sos["n_games_remaining"] if sos else None,
    )

    return dict(
        team_id=team_id, name=team["name"], abbreviation=team["abbreviation"],
        synopsis=TEAM_SYNOPSES.get(team["abbreviation"], ""),
        season=season,
        metrics=dict(
            ops=metric("ops", "ops_30d"),
            era=metric("era", "era_30d"),
            bullpen_era=metric("bullpen_era"),   # no 30d - see module docstring
            fip=metric("fip", "fip_30d"),
            baserunning_value=metric("baserunning_value"),
            remaining_sos=sos_metric,   # 30 = hardest remaining schedule - see strength_of_schedule.py
            war=metric("war"),
            il_count=metric("il_count"),
            park_factor=metric("park_factor"),
        ),
    )
