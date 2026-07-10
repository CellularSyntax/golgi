# IT'IS tissue-properties database (user-provided)

golgi uses the **IT'IS Foundation Material Parameter Database** for the Cole–Cole
tissue dielectric/conductivity presets. This database is **not redistributed with
golgi** — it is freely available from the IT'IS Foundation under their own license
terms, so you fetch it yourself.

## Get it — the easy way

From the repository root:

```bash
python -m golgi.conductivity.fetch_itis      # or:  golgi fetch-tissue-db
```

This downloads the database **directly from itis.swiss** (golgi does not host or
redistribute it) and installs it here as `IT'IS_Material_database_V4.1.db`. golgi
picks it up automatically on the next start.

## Get it — manually

1. Download the database from the IT'IS Foundation:
   <https://itis.swiss/virtual-population/tissue-properties/>
2. Put a SQLite file in this folder named either
   `IT'IS_Material_database_V4.2.db` or `IT'IS_Material_database_V4.1.db`
   (or point `$GOLGI_ITIS_DB` at any path).

## Without it

golgi still runs — the Cole–Cole dialog falls back to the **Custom** preset and you
can enter conductivities by hand. A one-time hint at startup reminds you.

## Attribution

The downloaded database is the **IT'IS Foundation Tissue Properties Database**:

- **Title:** Tissue Properties Database V4-1
- **Creator / Publisher:** IT'IS Foundation
- **Release date:** 2022-02-22
- **DOI:** [10.13099/VIP21000-04-1](https://doi.org/10.13099/VIP21000-04-1)

Please cite it in any work that uses golgi's tissue properties, and observe the
IT'IS Foundation's license/terms.
