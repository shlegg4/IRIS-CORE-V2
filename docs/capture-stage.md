# Capture stage

The Media Foundation reader records source and host-arrival timestamps before handing immutable samples to a capacity-one latest-sample channel. `CaptureClock` maps the stream clock into `steady_clock`, estimates drift, rejects regressions and reports residual error. The decoder writes into preallocated GPU slots and attaches a CUDA event, so downstream work can wait on readiness without synchronising the capture stream.

The live default is `DropOldest`: freshness is preferred and every discarded sample is counted. Exact format negotiation is required unless `allow_format_fallback` is explicitly enabled.

Use `iris_list_cameras` to inspect stable symbolic links and `iris_capture_probe [frame-count]` for hardware validation.
