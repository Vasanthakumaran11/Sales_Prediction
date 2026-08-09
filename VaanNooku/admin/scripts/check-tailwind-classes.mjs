#!/usr/bin/env node
/**
 * Guardrail against invalid Tailwind utility tokens (e.g. text-slate-850, w-58,
 * duration-205, border-3) which fail SILENTLY: Tailwind just skips generating
 * a rule for them, so the element renders with that property unstyled and
 * nothing errors at build time. This script catches them at lint time instead.
 *
 * Run via `npm run lint` (wired in package.json) or directly:
 *   node scripts/check-tailwind-classes.mjs
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC_DIR = path.join(__dirname, "..", "src");

const VALID_SHADES = new Set([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]);
const VALID_BORDER_WIDTHS = new Set([0, 1, 2, 4, 8]);
const VALID_DURATIONS = new Set([0, 75, 100, 150, 200, 300, 500, 700, 1000]);
const VALID_SPACING = new Set([
  0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24,
  28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96,
]);
const COLOR_NAMES =
  "slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose";

// className="..." and backtick template literals — good enough for this codebase's patterns
const CLASS_ATTR_RE = /className\s*=\s*(?:"([^"]*)"|'([^']*)'|`([^`]*)`)/g;

function findAllFiles(dir, exts, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      findAllFiles(full, exts, out);
    } else if (exts.some((e) => entry.name.endsWith(e))) {
      out.push(full);
    }
  }
  return out;
}

function checkClassString(str, violations, file, lineNo) {
  for (const token of str.split(/\s+/)) {
    if (!token) continue;

    // Arbitrary values (w-[240px], text-[9px], bg-[url(...)]) are always fine — skip.
    if (token.includes("[")) continue;

    let m;

    if ((m = token.match(new RegExp(`^(?:hover:|focus:|dark:|sm:|md:|lg:)*(?:bg|text|border|ring|divide|from|to|via|placeholder|accent|caret|outline|decoration|fill|stroke)-(?:${COLOR_NAMES})-(\\d+)(?:/\\d+)?$`)))) {
      const shade = parseInt(m[1], 10);
      if (!VALID_SHADES.has(shade)) {
        violations.push(`${file}:${lineNo}: invalid color shade "${token}" (nearest valid: ${nearest(shade, VALID_SHADES)})`);
      }
      continue;
    }

    if ((m = token.match(/^(?:hover:|focus:|dark:)*border(?:-[trbl])?-(\d+)$/))) {
      const w = parseInt(m[1], 10);
      if (!VALID_BORDER_WIDTHS.has(w)) {
        violations.push(`${file}:${lineNo}: invalid border width "${token}" (nearest valid: ${nearest(w, VALID_BORDER_WIDTHS)})`);
      }
      continue;
    }

    if ((m = token.match(/^duration-(\d+)$/))) {
      const d = parseInt(m[1], 10);
      if (!VALID_DURATIONS.has(d)) {
        violations.push(`${file}:${lineNo}: invalid duration "${token}" (nearest valid: ${nearest(d, VALID_DURATIONS)})`);
      }
      continue;
    }

    if ((m = token.match(/^(?:hover:|focus:|sm:|md:|lg:)*[wh]-(\d+(?:\.\d+)?)$/))) {
      const n = parseFloat(m[1]);
      if (!VALID_SPACING.has(n)) {
        violations.push(`${file}:${lineNo}: invalid width/height scale "${token}" (nearest valid: ${nearest(n, VALID_SPACING)}, or use an arbitrary value like w-[${n * 0.25}rem])`);
      }
      continue;
    }
  }
}

function nearest(value, validSet) {
  let best = null;
  let bestDiff = Infinity;
  for (const v of validSet) {
    const diff = Math.abs(v - value);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = v;
    }
  }
  return best;
}

const files = findAllFiles(SRC_DIR, [".js", ".jsx", ".ts", ".tsx"]);
const violations = [];

for (const file of files) {
  const content = readFileSync(file, "utf8");
  const lines = content.split("\n");
  let match;
  CLASS_ATTR_RE.lastIndex = 0;
  while ((match = CLASS_ATTR_RE.exec(content))) {
    const classStr = match[1] ?? match[2] ?? match[3] ?? "";
    const upTo = content.slice(0, match.index);
    const lineNo = upTo.split("\n").length;
    checkClassString(classStr, violations, path.relative(process.cwd(), file), lineNo);
  }
}

if (violations.length > 0) {
  console.error(`\nFound ${violations.length} invalid Tailwind utility token(s):\n`);
  for (const v of violations) console.error("  " + v);
  console.error("\nThese classes don't exist in Tailwind's default scale and render with NO style applied.\n");
  process.exit(1);
} else {
  console.log("No invalid Tailwind utility tokens found.");
}
