# EdgeMem UI manual checklist

Open `edgemem/ui/index.html` from disk or serve it statically. The defaults match `python scripts/run_all.py`: Device A `http://127.0.0.1:8001`, Device B `http://127.0.0.1:8002`, cloud `http://127.0.0.1:9000`.

1. **Configure endpoints.** Change a service URL, select **Apply**, and confirm the dashboard refreshes; restore the defaults afterward. API equivalent: GET `/health` from A and B and GET `/state` from the cloud URL.
2. **Inspect both device headers.** Confirm device IDs, online/offline state, point totals, pending count, and last-sync timestamps are shown. API equivalent: GET `/health` and `/sync-status` from each device.
3. **Toggle connectivity.** Select **Go online**, then **Go offline**; verify the status pill follows each action. API equivalent: POST `/network` with both `{"online":true}` and `{"online":false}`, then GET `/health`.
4. **Run a hybrid search.** Search A for a paraphrase such as `shaking and oil dripping`; confirm results show text, equipment, severity, code, and dense/sparse/fused scores. API equivalent: POST `/search` with `mode: "hybrid"`.
5. **Check filters and modes.** Search with equipment and minimum severity filters, then try Dense and Sparse modes. API equivalent: POST `/search` with `filters: {"equipment":"compressor","severity_gte":"high"}` and with each mode.
6. **Log an observation offline.** Create a new record on A while it is offline; confirm A can search it and pending increases. API equivalent: POST `/log`, GET `/sync-status`, and POST `/search` on A.
7. **Verify offline sync is a no-op.** Click **Sync now** while A is offline; pending must remain unchanged and the result should say `offline`. API equivalent: POST `/sync-now` and GET `/sync-status`.
8. **Sync A to cloud.** Bring A online and synchronize; confirm pending clears and cloud sequence advances. API equivalent: POST `/network`, POST `/sync-now`, and GET cloud `/state`.
9. **Pull to B.** Bring B online, synchronize, then search B for the new observation; confirm its ID and payload match A. API equivalent: POST `/network`, POST `/sync-now`, then POST `/search` and GET `/state` on B.
10. **Edit, tombstone, and converge.** Edit the new record's severity, sync, then delete it and sync again; confirm it disappears from search while remaining as a tombstone in state, and the convergence badge returns to green when A, B, and cloud hashes agree. API equivalent: PATCH `/points/{id}`, DELETE `/points/{id}`, sync both devices, GET all three `/state` responses, and compare their canonical payload maps.

The API-equivalent run is automated only for verification during development; these steps remain a manual browser checklist for future changes.
