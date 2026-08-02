---
title: "zotero-smart-ingest: An Agent Skill That Puts Papers in Zotero (With PDFs, No Duplicates)"
date: 2026-08-01T10:00:00-04:00
categories:
  - Tutorial
tags:
  - zotero
  - agent-skills
  - ai-agent
  - arxiv
  - google-scholar
  - open-source
  - python
  - research-workflow
---

I read a lot of finance papers. My workflow used to be: find paper → try to download PDF from five different places → manually create a Zotero entry → attach the PDF → realize I already had it from last year. Painful, manual, and surprisingly hard to automate — because most "paper" tooling assumes the published version is the only version, and Google Scholar has no API.

So I built [zotero-smart-ingest](https://github.com/linxichen/zotero-smart-ingest), an agent skill that turns "I want this paper in Zotero" into a verified library entry with a PDF — while a human picks the best version when the article exists in several forms. This post covers what it does, the gnarly bits (Scholar scraping, Zotero's v3 API, bot challenges), and what I learned.

## The pipeline

```
query (title / DOI / arXiv ID)
  → search: arXiv + Crossref + Google Scholar + Unpaywall + Semantic Scholar
  → group versions of the same article (by DOI, else normalized title)
  → dedup against your Zotero library (DOI or fuzzy title match)
  → human picks the best version from a numbered menu
  → download PDF, verify %PDF- magic bytes
  → create Zotero item + attach PDF (v3 three-step upload)
  → read back and confirm
```

Everything is stdlib-only Python, no API keys required (Unpaywall just wants an email). Install with:

```bash
npx skills add linxichen/zotero-smart-ingest
```

## Why "best version" matters

The same paper exists as an arXiv preprint, a published journal article, an author's homepage copy, and an SSRN working paper. They're not interchangeable: for citation you want the published version with the DOI; for reading you might want the arXiv PDF. The skill searches everywhere, groups versions of the same article together (Crossref's `container-title` filter is great for pinning down the published DOI), and makes the human pick. Nobody auto-picks a version for you.

## Google Scholar without an API

Scholar has no public API and blocks datacenter IPs. The skill handles this with a three-tier escalation:

1. **Polite HTTP scraping** — browser User-Agent + consent cookie, version-cluster expansion to enumerate every version of an article (journal, preprint, working paper, author copy).
2. **Chrome CDP** — when the scrape is blocked, it drives a real Chrome via the DevTools Protocol (stdlib-only WebSocket client, no dependencies). Same candidates, plus the `[PDF]` links Scholar shows and BibTeX pulled through each result's cite popup. On macOS, Chrome 149+ refuses the debug port on the default profile dir — the trick is copying `Default` to `~/.hermes/chrome-data` and launching with `--user-data-dir` (re-copy to refresh cookies).
3. **Semantic Scholar API** — the free open-API safety net, with retry-on-429 and an optional `S2_API_KEY`.

Scholar also A/B-tests its HTML. Mid-session it flipped to a new layout (`gs_r gs_or gs_scl` blocks, a JS-driven cite button, extra attributes on the PDF link div) and silently broke my parser. Lesson: treat Scholar markup as unstable — split result blocks by lookahead, not fixed closing tags, and always sanity-filter results by title-token overlap with the query.

## Zotero v3 API gotchas

Two things the docs make you discover the hard way:

- **Item creation wants a bare JSON array** (`[{...}]`), not `{"items": [...]}` — the wrapper gets rejected with "Uploaded data must be a JSON array".
- **PDF upload is a three-step flow**, not the old single multipart POST (which fails with "POST data not provided"): create the attachment item → authorize with `md5&filename&filesize&mtime` and `If-None-Match: *` → POST `prefix + file + suffix` to the returned storage URL → register the upload. All three steps plus read-back verification are in the skill.

## Library dedup

Before the menu even shows, the skill fetches your library once and flags candidates you already have — matched by DOI, else by fuzzy title (token overlap ≥80%, so "The Probability of Backtest Overfitting" matches "Probability of Backtest Overfitting"). If everything found is already there, it stops with "nothing to ingest". In our first live test, it correctly caught that I already had Hansen's SPA test paper and skipped it.

## The PDF-hunting playbook

Unpaywall's open-access links are great until they're not — Wiley's "bronze" PDFs 403 bots. When the obvious path fails, in order: Semantic Scholar's `openAccessPdf` field → Scholar's `[PDF]` links → author homepages → institutional repositories → the Wayback Machine. Two war stories: the UNC Carolina Digital Repository hosts a postprint of Hansen (2005) but serves a JS bot challenge on live downloads — the Wayback snapshot of the same URL bypasses it entirely; and Lo's "The Statistics of Sharpe Ratios" PDF still lives on the archived MIT site as `faj2.pdf`, discoverable via the CDX API.

## Tested live

Six classic quant papers, six links, one command each:

- Hansen's *A Test for Superior Predictive Ability* — already in the library, dedup flagged it, nothing ingested
- Lo's *The Statistics of Sharpe Ratios*, Bailey et al.'s *The Probability of Backtest Overfitting*, Sullivan-Timmermann-White, White's *A Reality Check for Data Snooping*, Harvey & Liu's *Backtesting* — all ingested, all with PDFs (author site, LSE eprints, course page, Duke, Wayback), all tagged `quant`

## Lessons

- Verify example DOIs via Crossref before putting them in docs — my first "canonical" DOI was a completely different paper.
- Wayback CDX (`web.archive.org/cdx/search/cdx`) is an underrated PDF search engine.
- If a paper's not on the first three sources, it's still somewhere — institutional repos and course pages are full of postprints.
- 1Password + `op run` keeps API keys out of every log.
- Human-in-the-loop is a feature: version selection is a judgment call, and the skill's job is to make it a 10-second menu pick instead of a 20-minute hunt.

The repo is open source — PRs welcome, especially for new sources.
