---
title: "Chart Gallery (regression page)"
date: 2026-08-30
tags: [meta]
draft: true
summary: Edge-case fixtures for assets/chart.js — open after any chart change; everything must render and the console must stay clean.
---

Dev page, not published. After **any** change to `assets/chart.js` or its CSS,
open this page in the preview and check: every chart renders, labels don't
collide, hover works on each, the console has no warnings, and both themes
look right. Then re-check the real figures in W260830.

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="single">single series</div>

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="four">four series</div>

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="dense">dense, no markers</div>

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="negative">negative values</div>

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="collide">end-label collision</div>

Options override via embed attribute (markers forced off on sparse data):

<div class="chart" data-src="writings/figures/chart-fixtures.json" data-metric="single" data-opt-markers="false" data-opt-data-table="false">options override</div>
