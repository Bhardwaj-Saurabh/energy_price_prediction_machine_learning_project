"""Monitoring pipeline — compute drift signals and emit a retrain verdict.

Reads load forecasts (predictions), load observations (truth), and the
feature matrix (current vs. baseline distributions); computes rolling
MAPE per zone and per-feature PSI; consults the
:func:`energy_forecaster.domain.rules.retrain.should_retrain` rule;
emits a :class:`MonitoringResult`. The orchestrator (cron / Prefect)
acts on the verdict — this pipeline does not retrain.
"""
