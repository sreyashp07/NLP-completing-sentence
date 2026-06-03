"""
Design system color constants.
Single source of truth for UI colors across the application.
"""

# Background colors
BG_PRIMARY = "#0A0E1A"
BG_SURFACE = "#131A2A"
BG_ELEVATED = "#1A2332"
BG_BORDER = "#1F2A3D"

# Text colors
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#DDE3EC"
TEXT_TERTIARY = "#A8B5C8"
TEXT_MUTED = "#6B7A94"

# Accent colors
ACCENT_PRIMARY = "#4086F5"
ACCENT_HOVER = "#2E6DD9"
ACCENT_SOFT = "#8AB4F8"

# Status colors (muted, professional)
STATUS_CRITICAL = "#E63946"
STATUS_HIGH = "#F4A261"
STATUS_MEDIUM = "#2A9D8F"
STATUS_LOW = "#264653"

# Priority color mapping
PRIORITY_COLORS = {
    "critical": STATUS_CRITICAL,
    "high":     STATUS_HIGH,
    "medium":   STATUS_MEDIUM,
    "low":      STATUS_LOW,
}

# Intent color palette (blue gradient)
INTENT_PALETTE = [
    "#4086F5", "#5B9BF8", "#2E6DD9", "#1F4FA8",
    "#7AB1FB", "#3A78E0", "#1A3D80", "#5181E5", "#2553B8",
]
