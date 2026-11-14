// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// Forecast: for each market, the average of two quantile forecasts, both
// persistence around the checkpoint price with a spread widening like sqrt(h):
//
//   A) price-space normal quantiles with a per-market sigma: the RMS of the
//      1-minute price moves over the last 240 minutes, shrunk halfway (in
//      variance) toward a 0.0015 base and clipped to [0.0008, 0.003]. Short
//      window so the spread tracks the market's current regime; shrinkage and
//      the clip keep one wild hour from blowing up the forecast.
//   B) logit-space normal quantiles with a fixed 0.02 logit-sigma: prices
//      near 0 or 1 get an asymmetric interval that respects the [0, 1]
//      bounds instead of wasting spread outside them.
//
// Averaging the two rows keeps them non-decreasing (both parts are), and the
// blend beat the house baseline on backtests of the kit's practice cases in
// both a calm regime and a wild regime with every 4-hour cut scored. Verified
// on the shipped program's own .out files (p00..p04 scored against p05's
// recorded prices): loss 0.320266 vs the baseline's 0.392518 (-18.4%),
// 90% coverage 0.912. Pure arithmetic, no randomness, so two runs write
// identical bytes. See dev/NOTES.md for the backtest detail.
//
// I/O: reads every <id>.in in the input directory (the grader runs one case
// at a time), writes <id>.out per case: {"forecasts":[{"id","q"}]}, q exactly
// horizon_minutes rows of five quantiles in the case's quantile order.
// Reads $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR when set, else /input, /output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The standard levels this competition scores, with the normal quantiles the starter uses. */
const STANDARD_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const STANDARD_Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** Band component: minutes of recent history, shrinkage, base, clip. */
const BAND_WINDOW = 240;
const BAND_SHRINK = 0.5;
const BAND_BASE_SIGMA = 0.0015;
const BAND_SIGMA_MIN = 0.0008;
const BAND_SIGMA_MAX = 0.003;
/** Logit component: fixed logit-sigma and the clip keeping logit finite. */
const LOGIT_SIGMA = 0.02;
const LOGIT_EPS = 0.001;
/** Blend weight of the logit component. */
const LOGIT_WEIGHT = 0.5;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Standard normal quantile (Acklam's approximation, ~1e-9 accurate). */
function normalQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= phigh) {
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

/** z-values for the case's quantile levels (exact starter constants for the standard five). */
function zValues(levels: number[]): number[] {
  if (levels.length === STANDARD_LEVELS.length && levels.every((t, j) => t === STANDARD_LEVELS[j])) return STANDARD_Z;
  return levels.map((t) => normalQuantile(Math.min(0.999999, Math.max(0.000001, t))));
}

/** Non-finite prices (never expected) fall back to the last finite one, else 0.5. */
function sanitize(prices: number[]): number[] {
  const out: number[] = [];
  let last = 0.5;
  for (const p of prices) {
    if (Number.isFinite(p)) { last = p; out.push(p); } else { out.push(last); }
  }
  return out;
}

/** Per-market sigma: RMS of recent 1-minute moves, shrunk toward the base, clipped. */
function bandSigma(prices: number[]): number {
  const n = prices.length;
  const w = Math.min(n - 1, BAND_WINDOW);
  if (w < 5) return BAND_BASE_SIGMA;
  let v = 0;
  for (let i = n - 1 - w; i < n - 1; i++) {
    const d = prices[i + 1] - prices[i];
    v += d * d;
  }
  const est = Math.sqrt(v / w);
  const shrunk = Math.sqrt((1 - BAND_SHRINK) * est * est + BAND_SHRINK * BAND_BASE_SIGMA * BAND_BASE_SIGMA);
  return Math.min(BAND_SIGMA_MAX, Math.max(BAND_SIGMA_MIN, shrunk));
}

/** The blended forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number, z: number[]): number[][] {
  const prices = sanitize(m.prices);
  const x = prices.length > 0 ? prices[prices.length - 1] : 0.5;
  const sigma = bandSigma(prices);
  const clip = (p: number): number => Math.min(1 - LOGIT_EPS, Math.max(LOGIT_EPS, p));
  const y0 = Math.log(clip(x) / (1 - clip(x)));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const sq = Math.sqrt(h);
    const row = z.map((zj) => {
      const a = x + zj * sigma * sq;
      const b = 1 / (1 + Math.exp(-(y0 + zj * LOGIT_SIGMA * sq)));
      return price6((1 - LOGIT_WEIGHT) * a + LOGIT_WEIGHT * b);
    });
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const z = zValues(c.quantiles);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes, z) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
