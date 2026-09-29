import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import prettier from "prettier";
import { readFileSync, writeFileSync } from "node:fs";

// Enforce changed supported files; the untouched historical tree has no baseline yet.
const tracked = execFileSync(
  "git",
  ["diff", "--name-only", "--diff-filter=ACMR", process.env.FORMAT_BASE || "HEAD", "--"],
  { encoding: "utf8" },
);
const untracked = execFileSync("git", ["ls-files", "--others", "--exclude-standard"], {
  encoding: "utf8",
});
const files = [...new Set((tracked + untracked).trim().split("\n"))].filter(
  (file) => existsSync(file) && /\.(md|json|ya?ml|mjs)$/.test(file),
);
let failed = false;
for (const file of files) {
  const info = await prettier.getFileInfo(file, { ignorePath: ".prettierignore" });
  if (info.ignored || !info.inferredParser) continue;
  const input = readFileSync(file, "utf8");
  const options = { ...(await prettier.resolveConfig(file)), filepath: file };
  if (process.argv.includes("--write")) writeFileSync(file, await prettier.format(input, options));
  else if (!(await prettier.check(input, options))) {
    console.error(`Formatting required: ${file}`);
    failed = true;
  }
}
process.exitCode = failed ? 1 : 0;
