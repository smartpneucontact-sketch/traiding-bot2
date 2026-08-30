FROM python:3.11-slim

WORKDIR /app

# combo_v2 is a pure-rule strategy — no native ML libraries needed.
# numpy/pandas/scipy wheels come pre-built on PyPI.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py .
COPY dashboard.py .
COPY core/ core/
COPY model/ model/
# Pre-registered forward-test protocols: the runner binds each slot's
# protocol sha at its first funded rebalance — these files MUST be in the
# image (their absence made the 2026-07-20 primary bind fail silently).
COPY protocol.json protocol_exp.json protocol_process.json PROTOCOL.md ./
# Spec manifest + program calendar are runtime inputs (core/invariants.py,
# dashboard calendar tiles). Their absence from the image is exactly the
# fault class the invariant checker flags — its first in-container run
# caught this very line missing (2026-08-30).
COPY spec_manifest.json program_calendar.json OPS_LOG.md ./

# Persistent data dir (mount a Railway volume here for state + logs)
RUN mkdir -p /app/data/state /app/data/logs
ENV DATA_DIR=/app/data

EXPOSE 8080

# Run dashboard (includes APScheduler cron + Flask web UI)
CMD ["python", "dashboard.py"]
