# Architecture

`inspect_dataset.py` is deliberately the first stage. The API layer is thin; domain calculations are deterministic and provider/model adapters are replaceable. Real deployments should add PostgreSQL/SQLAlchemy repositories, Redis-backed event fan-out, authentication middleware, durable alert acknowledgement and a real notification adapter before production exposure.

Pipeline: source adapter → YOLO adapter → tracker → temporal verifier → event manager → visual growth/spread proxies → environmental provider → risk/early warning → alert/notification → persistence → WebSocket/dashboard.
