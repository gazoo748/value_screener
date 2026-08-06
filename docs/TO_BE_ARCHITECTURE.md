# Target Architecture

## Scope

The public value screener will be ported to Rust and will consume primary or
raw market data rather than depend on a curated data vendor as its system of
record. Vendor feeds remain useful inputs and enrichment sources, but must not
define the application’s durable identity model.

## Instrument identity model

Do not use a ticker symbol as a primary key. A ticker identifies a listing only
for a period of time and can change or be reused.

Use an immutable application-generated `security_id` as the primary key, with
time-bounded identifiers and listings associated to it:

```text
security
  security_id

security_identifier
  security_id
  scheme              # FIGI, COMPOSITE_FIGI, SHARE_CLASS_FIGI, CUSIP, ISIN,
                      # vendor_permaticker, etc.
  value
  valid_from
  valid_to
  source
  confidence

listing
  listing_id
  security_id
  mic / exchange
  ticker
  valid_from
  valid_to
```

The `security_identifier` table should enforce uniqueness for an active
`(scheme, value)` mapping where the identifier standard guarantees uniqueness,
while retaining historical records rather than overwriting them. The `listing`
table records ticker and venue changes independently from the instrument.

## FIGI and Sharadar permaticker

FIGI and Sharadar’s `permaticker` serve a similar purpose: both are more
durable than a ticker symbol. They are not interchangeable.

- `permaticker` is a Sharadar-specific identifier. Preserve it as a historical
  vendor alias, but do not make the new platform depend on it.
- FIGI is an external standardized identifier and is useful as a bridge across
  data sources.
- FIGI has distinct instrument, share-class, and composite forms. Store the
  form explicitly in `scheme`; do not collapse them into one field.
- A normal ticker-symbol change does not change the FIGI. A genuinely new
  instrument receives a new FIGI.

The existing `sharadar.tickers.figi` field is nullable enrichment data. Its
non-unique index supports lookup, but the Rust application must resolve it to
the internal `security_id` through `security_identifier`.

## Historical mapping strategy

Begin recording identifier and listing deltas from every ingested source now.
For every observed change, close the prior row’s validity interval and create a
new row with source and confidence metadata.

Do not fabricate history where no point-in-time source exists. For historical
backfill, prefer an authoritative source’s archived security-master snapshots.
Use current FIGI mappings only as enrichment or validation, not as proof of a
past ticker-to-identifier relationship.

## Data-ingestion implications

1. Normalize every source record to an internal security or listing candidate.
2. Resolve by stable identifiers first, then by venue, ticker, and effective
   date; retain ambiguous matches for review rather than guessing.
3. Store raw source payloads and provenance separately from normalized tables.
4. Version all mapping decisions so corrections are reproducible.
5. Expose instrument identity to application queries through stable internal
   IDs; ticker is presentation and search metadata.

## Published-assessment eligibility

The public application must distinguish an adverse assessment from an
unavailable assessment. Missing observations, a newly listed instrument, or a
metric that does not apply to the instrument must never be silently converted
into a favorable or unfavorable score.

The assessment controller produces a versioned, as-of-date `published_assessment`
for each listing. The UI reads that record and its evidence; it must not infer
grades from missing source fields or recompute them in templates.

### Risk-rating eligibility

- Publish an A–F market-risk grade only when all required inputs for that
  instrument type meet their minimum-history requirements.
- For an ETF, require sufficient price history to calculate volatility,
  drawdown, and beta (currently 252 paired trading observations). Until then,
  publish `risk_status = provisional_insufficient_history`, not a letter grade.
- Keep available observations, such as the ETF share’s average dollar volume,
  as evidence. Do not call the highest available observation the “primary
  risk” when other required observations are missing.
- Model ETF liquidity separately from underlying-portfolio liquidity. ETF
  share dollar volume is one execution signal; bid/ask spread, creation/
  redemption capacity, assets under management, and holdings liquidity are
  separate signals when data is available.

### Income-quality eligibility

- Publish an income-quality letter grade only after the instrument has a
  sufficient, expected distribution history. The threshold must be explicit
  and strategy-aware (for example, at least 12 months of history and enough
  observations for the stated distribution frequency).
- Until that threshold is met, publish
  `income_quality_status = unrated_insufficient_distribution_history`, not a
  D or “poor” rating.
- A missing equity-style coverage field is `not_applicable` for an ETF; it is
  not weak coverage. ETF distribution analysis must use inputs appropriate to
  its strategy, such as distribution consistency, source of distribution when
  available, total-return support, and return-of-capital treatment.
- Preserve raw distribution records and the expected-frequency assumption in
  the assessment evidence so a user can see why the status is provisional or
  unrated.

### Presentation rules

- Display the status label, the as-of date, the unmet requirement, and the
  available evidence together. Example: “Provisional — 248 of 252 required
  trading observations.”
- Do not rank, recommend, or use an ungraded/provisional metric in a composite
  score, default sort, or marketing claim.
- Retain the assessment version, input timestamps, thresholds, and calculation
  version so any displayed conclusion is reproducible.

### Screener navigation state

- Preserve the user’s screener state when moving from the results table to a
  security-detail page and back. State includes server-side filters, result
  limit, selected table filters, sort columns, and sort directions.
- Support ordered multi-column sorting as a serializable state value so a
  primary sort and secondary sorts are restored exactly, rather than merely
  returning to the default sort.
- Treat this state as navigation/UI state, not part of the published
  assessment. It may be encoded in a bounded URL query, signed server-side
  session, or another shareable state mechanism appropriate to the Rust web
  application.
