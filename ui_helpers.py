"""
Presentation-only helpers for the Streamlit player page: colored percentile
bars, team logos, and the stylized bat-swing/field visual. No data logic
lives here - see player_page.py / percentiles.py for that.

All custom HTML sets explicit text colors rather than relying on inherited
theme colors - Streamlit's own default text color can be light (dark-theme
default), which silently makes text invisible against a light background
injected via custom CSS.
"""
import math

import pandas as pd

_BLUE = (37, 99, 235)
_GREY = (156, 163, 175)
_RED = (220, 38, 38)
TEXT_DARK = "#1a1a1a"
TEXT_MUTED = "#666666"


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gradient_color(pct):
    """0 -> blue, 50 -> grey, 100 -> red. Two-segment interpolation so the
    midpoint is a clean neutral grey rather than a muddy blue/red blend."""
    pct = max(0, min(100, pct))
    if pct <= 50:
        r, g, b = _lerp(_BLUE, _GREY, pct / 50)
    else:
        r, g, b = _lerp(_GREY, _RED, (pct - 50) / 50)
    return f"rgb({r},{g},{b})"


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _flatten(html):
    """
    Collapses a multi-line, indented HTML string to a single line with no
    leading whitespace. Streamlit's markdown renderer treats 4+ spaces of
    indentation at the start of a line as a CommonMark indented code block -
    when several of these multi-line f-strings get concatenated, some
    boundaries end up looking like one to the parser, and that block renders
    as literal escaped text instead of HTML. Flattening avoids the ambiguity
    entirely rather than relying on getting the whitespace exactly right.
    """
    return " ".join(line.strip() for line in html.strip().splitlines())


def percentile_bar_html(label, raw_value, pct, value_fmt="{:.1f}"):
    color = gradient_color(pct)
    value_str = value_fmt.format(raw_value) if pd.notna(raw_value) else "-"
    return _flatten(f"""
    <div style="margin-bottom:10px; max-width:340px;">
      <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:3px; color:{TEXT_DARK}; font-weight:600;">
        <span>{label}</span>
        <span>{value_str}&nbsp;&nbsp;({pct:.0f})</span>
      </div>
      <div style="background:#e5e7eb; border-radius:6px; height:16px; width:100%;">
        <div style="background:{color}; width:{pct:.0f}%; height:16px; border-radius:6px;"></div>
      </div>
    </div>
    """)


def percentile_group_header_html(title):
    return (f'<div style="font-size:12px; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:0.5px; color:{TEXT_MUTED}; margin:12px 0 6px 0;">{title}</div>')


def team_logo_url(team_id):
    return f"https://www.mlbstatic.com/team-logos/{team_id}.svg" if team_id else ""


def player_card_html(name, team_id, stat_pairs, badge_html=""):
    """
    Horizontal card spanning the full width: logo + name on the left, then
    each (label, value) in stat_pairs laid out as its own block, then an
    optional badge (e.g. Stuff+) on the far right.
    """
    logo = team_logo_url(team_id)
    logo_html = (f'<img src="{logo}" style="width:44px; height:44px;" '
                 f'onerror="this.style.display=\'none\'"/>') if logo else ""

    stat_blocks = "".join(
        f'<div style="text-align:center; padding:0 14px;">'
        f'<div style="font-size:11px; color:{TEXT_MUTED};">{label}</div>'
        f'<div style="font-size:18px; font-weight:700; color:{TEXT_DARK};">{value}</div>'
        f'</div>'
        for label, value in stat_pairs
    )

    return _flatten(f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:10px 18px; background:#fff;
                display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
      <div style="display:flex; align-items:center; gap:10px; min-width:160px;">
        {logo_html}
        <div style="font-size:20px; font-weight:800; color:{TEXT_DARK};">{name}</div>
      </div>
      <div style="display:flex; flex-wrap:wrap;">
        {stat_blocks}
      </div>
      <div>{badge_html}</div>
    </div>
    """)


def archetype_badge_html(archetype, description=""):
    return _flatten(f"""
    <div style="text-align:center; border:2px solid #444; border-radius:10px; padding:4px 12px;" title="{description}">
      <div style="font-size:10px; color:{TEXT_MUTED};">ARCHETYPE</div>
      <div style="font-size:18px; font-weight:800; color:{TEXT_DARK};">{archetype}</div>
    </div>
    """)


def stuff_plus_badge_html(value):
    if pd.isna(value):
        return f'<div style="font-size:12px; color:{TEXT_MUTED};">Stuff+ unavailable</div>'
    color = gradient_color(min(100, max(0, (value - 70) / 60 * 100)))
    return _flatten(f"""
    <div style="text-align:center; border:2px solid {color}; border-radius:10px; padding:4px 12px;">
      <div style="font-size:10px; color:{TEXT_MUTED};">STUFF+</div>
      <div style="font-size:24px; font-weight:800; color:{color};">{value:.0f}</div>
    </div>
    """)


def row_tint_styler(color_map, default_color="#999999", key_col="pitch_type"):
    """
    Returns a pandas Styler function (for df.style.apply(..., axis=1), called
    on a DataFrame where `key_col` is a real column, not the index) that
    tints every cell in the row with a faint version of that row's pitch
    color, looked up via color_map[row[key_col]].
    """
    def _style_row(row):
        color = color_map.get(row[key_col], default_color)
        faint = hex_to_rgba(color, 0.15) if color.startswith("#") else color
        return [f"background-color:{faint}"] * len(row)
    return _style_row


def _point(cx, cy, angle_deg, r):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def bat_swing_svg(lf_pct, cf_pct, rf_pct, bat_speed, tilt, attack_angle):
    """
    Field-level, umpire's-eye scene: a horizon-based perspective view (sky
    above, field below, three zones converging toward a vanishing point on
    the horizon - not a bird's-eye map) with a stylized bat-swing arc drawn
    across the foreground, as seen from directly behind the batter. The
    field zones are colored by this batter's actual share of batted-ball
    contact in that direction (red = more, blue = less). Illustrative, not a
    physics simulation of the swing.
    """
    vp_x, vp_y = 200, 95
    bottom_y = 300
    x0, x1, x2, x3 = 0, 400 / 3, 800 / 3, 400

    zones = [("LF", lf_pct, x0, x1), ("CF", cf_pct, x1, x2), ("RF", rf_pct, x2, x3)]
    field_html = ""
    for label, pct, xa, xb in zones:
        color = gradient_color(pct)
        field_html += (f'<polygon points="{vp_x},{vp_y} {xa:.0f},{bottom_y} {xb:.0f},{bottom_y}" '
                       f'fill="{color}" opacity="0.6" stroke="white" stroke-width="1"/>')
        mx = (xa + xb) / 2
        field_html += (f'<text x="{mx:.0f}" y="205" font-size="15" fill="white" '
                       f'text-anchor="middle" font-weight="bold" '
                       f'style="paint-order:stroke; stroke:#00000055; stroke-width:3px;">{label} {pct:.0f}%</text>')

    bat_speed_s = f"{bat_speed:.1f}" if pd.notna(bat_speed) else "-"
    tilt_s = f"{tilt:.0f}" if pd.notna(tilt) else "-"
    aa_s = f"{attack_angle:.0f}" if pd.notna(attack_angle) else "-"

    return _flatten(f"""
    <svg viewBox="0 0 400 300" width="100%" style="border-radius:8px;">
      <rect x="0" y="0" width="400" height="{vp_y}" fill="#bfe3f5"/>
      <rect x="0" y="{vp_y}" width="400" height="{bottom_y - vp_y}" fill="#5b8c3a"/>
      {field_html}
      <line x1="0" y1="{vp_y}" x2="400" y2="{vp_y}" stroke="#3a5f27" stroke-width="2"/>
      <ellipse cx="200" cy="272" rx="15" ry="24" fill="#222"/>
      <circle cx="200" cy="240" r="10" fill="#222"/>
      <path d="M 115,252 Q 200,145 295,238" stroke="#3b2415" stroke-width="7" fill="none" stroke-linecap="round"/>
      <circle cx="295" cy="238" r="5" fill="#c9a24b"/>
      <rect x="60" y="278" width="280" height="18" rx="9" fill="#00000066"/>
      <text x="200" y="291" font-size="12" text-anchor="middle" fill="white" font-weight="bold">
        Bat Speed: {bat_speed_s} mph | Tilt: {tilt_s}&#176; | Attack Angle: {aa_s}&#176;
      </text>
    </svg>
    """)
