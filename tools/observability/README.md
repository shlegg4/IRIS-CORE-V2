# IRIS developer observability

IRIS exposes Prometheus metrics on the loopback-only endpoint
`http://127.0.0.1:9464/metrics`. This stack stores those metrics in Prometheus and provisions an
IRIS dashboard in Grafana.

From the repository root:

```powershell
docker compose -f tools/observability/compose.yaml up -d
```

Then start `build/default/bin/iris_app.exe` and open:

- Grafana: <http://localhost:3000/d/iris-developer>
- Prometheus targets: <http://localhost:9090/targets>
- Raw IRIS metrics: <http://127.0.0.1:9464/metrics>

The local Grafana instance allows anonymous viewer access and binds only to loopback. Prometheus
retains seven days of development metrics in a Docker volume.

Stop the stack without deleting its history:

```powershell
docker compose -f tools/observability/compose.yaml down
```

Add `-v` only when you intentionally want to delete the Prometheus and Grafana volumes.
