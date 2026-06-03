# Changelog

## v1.1.0 - UI Redesign

### Changed
- Replaced all decorative emojis with professional typography
- New color palette: muted blues and professional accent colors
- Switched to Space Grotesk + IBM Plex Mono font stack
- Toned down hover effects for cleaner interactions
- Replaced bright green/yellow with muted teal/amber for status colors

### Fixed
- Session state key conflict on example button clicks
- Bar chart text position now outside for better visibility
- Streamlit config theme alignment with custom CSS

## v1.0.0 - Initial Production Release

### Added
- TF-IDF + Logistic Regression baseline model (9 classes)
- FastAPI REST API with Pydantic v2 validation
- Streamlit 4-page UI
- Docker containerization
- GitHub Actions CI/CD pipeline with 200+ tests
- MLflow experiment tracking
- Hugging Face Spaces deployment

### Performance
- 12ms inference latency (p50)
- 1.0 F1 on synthetic dataset
- 51MB model footprint
