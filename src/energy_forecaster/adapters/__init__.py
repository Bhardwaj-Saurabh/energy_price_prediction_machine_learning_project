"""Adapters layer — concrete implementations of application ports.

Every external dependency the application reaches outside itself for is
satisfied here by a concrete class. Adapter classes are the *only* place
in the codebase allowed to import third-party libraries (psycopg, mlflow,
feast, fastapi, azure SDKs, requests, …) — keeping those imports here
means the application and domain layers stay framework-agnostic.

Sub-packages mirror the ports they implement: ``adapters/clock/``
contains every implementation of :class:`Clock`, ``adapters/load_observation_repo/``
contains every implementation of :class:`LoadObservationRepository`, and
so on. Picking which adapter to wire in is the composition root's job —
adapters never reference each other.
"""
