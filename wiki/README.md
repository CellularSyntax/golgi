# golgi wiki — source

This folder holds the **source of the [GitHub wiki](https://github.com/CellularSyntax/golgi/wiki)** so
the documentation is version-controlled alongside the code. GitHub serves the wiki from a *separate*
git repository (`<repo>.wiki.git`); the files here are copied into it to publish.

> This `README.md` and `preview.html` are maintainer tools, **not** wiki pages — don't copy them to
> the wiki repo (copy only the other `*.md` files; the publish command below excludes them).

## Preview the wiki locally

You don't need GitHub (or a public repo) to check the wiki. `preview.html` is a self-contained viewer
that renders these pages with a sidebar, footer, working page-to-page links, and Mermaid diagrams —
close to how GitHub will show them. Serve this folder over HTTP and open it:

```bash
cd wiki
python3 -m http.server 8000
# then open http://localhost:8000/preview.html
```

(Use a local server, not a `file://` path — browsers block `fetch()` of local files. The viewer pulls
its renderer + Mermaid from a CDN, so it needs internet; everything else is local.) For a quick
single-page check with no server, VS Code's built-in **Markdown: Open Preview** (`⇧⌘V`) also works
(install *Markdown Preview Mermaid Support* for the diagrams).

## Pages

- `Home.md` — the wiki landing page.
- `_Sidebar.md` — navigation shown on every page.
- `_Footer.md` — footer shown on every page.
- Every other `*.md` — one content page each (the page name is the filename without `.md`; GitHub
  shows spaces for the hyphens).

Internal links use GitHub wiki page names, e.g. `[Installation](Installation)` and
`[Finite-Element Solver](Finite-Element-Solver)`.

## Publish / update the wiki

The wiki must have at least one page created in the GitHub UI before it can be cloned. Then:

```bash
# from the repository root
git clone https://github.com/CellularSyntax/golgi.wiki.git /tmp/golgi.wiki

# copy every page (but not the maintainer tools) into the wiki repo
rsync -av --exclude README.md --exclude preview.html wiki/ /tmp/golgi.wiki/

cd /tmp/golgi.wiki
git add -A
git commit -m "Update wiki from main repo wiki/ source"
git push
```

Re-run the `rsync` + commit + push whenever you edit a page here.

## Notes

- The `Home.md` logo uses a raw GitHub URL so it renders on the wiki even though `docs/` lives in the
  main repo, not the wiki repo.
- Mermaid diagrams render natively on github.com (no plugin needed).
- Keep page names stable — renaming a page breaks the inbound cross-page links and the sidebar.
