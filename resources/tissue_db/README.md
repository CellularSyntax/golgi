# IT'IS tissue-properties database (user-provided)

golgi uses the **IT'IS Foundation Material Parameter Database (V4.2)** for the
Cole–Cole tissue dielectric/conductivity presets. This database is **not
redistributed with golgi** — it is freely available from IT'IS under their own
license terms, so you download it yourself and drop it in this folder.

## Install it

1. Download the material database from the IT'IS Foundation:
   <https://itis.swiss/virtual-population/tissue-properties/> (free; subject to
   IT'IS's registration/attribution terms).
2. Place the SQLite file here so the path is exactly:

   ```
   resources/tissue_db/IT'IS_Material_database_V4.2.db
   ```

golgi loads it automatically on the next start; the Cole–Cole dialog and the
per-domain conductivity dropdowns then expose the full IT'IS tissue list.

## Without it

golgi still runs — the Cole–Cole dialog falls back to the **Custom** preset and
you can enter conductivities by hand. A one-time hint at startup reminds you
where to get the database.
