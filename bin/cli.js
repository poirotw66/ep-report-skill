#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

// Files and directories to copy from the package into the target location
const ASSETS = ["SKILL.md", "reference.md", "scripts"];

const SKILL_DIR_NAME = "ep-yf-report-skill";

function printUsage() {
  console.log(`
Usage:
  npx ep-yf-report-skill [target-dir]

Arguments:
  target-dir   Where to install the skill files.
               Defaults to .cursor/skills/${SKILL_DIR_NAME} inside the current working directory.

What this does:
  Copies SKILL.md, reference.md, and scripts/ into <target-dir>,
  then prints the absolute path of SKILL.md so you can add it to
  Cursor's agent_skills configuration.
`);
}

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const child of fs.readdirSync(src)) {
      copyRecursive(path.join(src, child), path.join(dest, child));
    }
  } else {
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
  }
}

function main() {
  const args = process.argv.slice(2);

  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    process.exit(0);
  }

  const cwd = process.cwd();
  const rawTarget = args[0]
    ? path.resolve(cwd, args[0])
    : path.join(cwd, ".cursor", "skills", SKILL_DIR_NAME);

  // Package root is two levels up from bin/cli.js
  const pkgRoot = path.resolve(__dirname, "..");

  console.log(`\nInstalling ep-yf-report-skill to:\n  ${rawTarget}\n`);

  for (const asset of ASSETS) {
    const src = path.join(pkgRoot, asset);
    const dest = path.join(rawTarget, asset);

    if (!fs.existsSync(src)) {
      console.warn(`  [skip] ${asset} not found in package, skipping.`);
      continue;
    }

    copyRecursive(src, dest);
    console.log(`  [ok]   ${asset}`);
  }

  const skillMdPath = path.join(rawTarget, "SKILL.md");

  console.log(`
Done! Add the following path to your Cursor agent_skills in settings:

  ${skillMdPath}

Example .cursor/settings.json entry:
  {
    "agent_skills": [
      { "fullPath": "${skillMdPath.replace(/\\/g, "\\\\")}" }
    ]
  }
`);
}

main();
