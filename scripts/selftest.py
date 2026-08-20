"""Self-test: render the app, drive the widgets, check the numbers.

    .venv/Scripts/python.exe scripts/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src import config, predictor, stats  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def new_app() -> AppTest:
    app = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()
    return app


def test_render() -> None:
    print("\n[render]")
    app = new_app()
    check(not app.exception, "app renders")
    check(len(app.number_input) == 9, f"{len(app.number_input)} ingredient inputs (with temperature)")
    check(
        any("PREDICT STRENGTH" in b.label for b in app.button),
        "predict button present",
    )
    check(len(app.tabs) == 2, f"{len(app.tabs)} tabs")


def test_family_switch() -> None:
    print("\n[feature set switch]")
    app = new_app()
    app.selectbox(key="family").set_value("no_temp").run()
    check(not app.exception, "switches to the 8-input family")
    check(len(app.number_input) == 8, f"{len(app.number_input)} ingredient inputs (no temperature)")


def test_every_model() -> None:
    print("\n[all six models]")
    values = {f.column: config.feature_median(f) for f in config.FEATURES}
    seen = 0
    for family in config.FAMILIES:
        for spec in predictor.available_models(family):
            prediction = predictor.predict(spec, values)
            ok = 0 < prediction < 150
            check(ok, f"{spec.name} ({family}) -> {prediction:.2f} MPa")
            seen += 1
    check(seen == 6, f"{seen} models served")


def test_prediction_is_recorded() -> None:
    print("\n[statistics are recorded and persist]")
    stats.reset()

    app = new_app()
    button = next(b for b in app.button if "PREDICT STRENGTH" in b.label)
    button.click().run()
    check(not app.exception, "prediction runs")

    payload = stats.load()
    check(payload["total"] == 1, f"one run recorded (total={payload['total']})")

    # A fresh process must see the same file.
    stats.record("gb_with_temp", "Gradient Boosting — Champion", "with_temp", 48.0)
    reloaded = stats.load()
    check(reloaded["total"] == 2, "counter survives a reload")

    rows = stats.per_model(reloaded)
    check(bool(rows), f"{len(rows)} model row(s) in the analytics table")
    check(
        all({"Model", "Runs", "Mean MPa"} <= set(row) for row in rows),
        "per-model columns present",
    )

    overall = stats.overall(reloaded)
    check(overall["total"] == 2, "overall total matches")
    stats.reset()
    check(stats.load()["total"] == 0, "reset clears the counters")


def test_out_of_range_flag() -> None:
    print("\n[out-of-range flag]")
    values = {f.column: config.feature_median(f) for f in config.FEATURES}
    values[config.CEMENT] = 5000.0
    flagged = predictor.out_of_range(values, config.WITH_TEMP_COLUMNS)
    check("Cement" in flagged, "extreme cement is flagged")


def main() -> int:
    test_render()
    test_family_switch()
    test_every_model()
    test_prediction_is_recorded()
    test_out_of_range_flag()
    print("\n" + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
