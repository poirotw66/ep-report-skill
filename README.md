# ep-yf-report-skill

> A **Cursor Agent Skill** that generates SVG stock charts and a detailed markdown report by extracting tickers from an episode transcript (`.md`), downloading recent prices from Yahoo Finance, and matching episode investment themes to market performance.

---

## Prerequisites

- **Python 3.9+** with the following packages:

```bash
pip install yfinance pandas matplotlib pyyaml
```

---

## Install the skill

Run this once inside any project where you want to use the skill:

```bash
npx ep-yf-report-skill
```

This copies `SKILL.md`, `reference.md`, and `scripts/` into:

```
<your-project>/.cursor/skills/ep-yf-report-skill/
```

You can also specify a custom target directory:

```bash
npx ep-yf-report-skill /path/to/my-skills/ep-yf-report-skill
```

---

## Add to Cursor

After running the command above, it will print a path like:

```
/your-project/.cursor/skills/ep-yf-report-skill/SKILL.md
```

Add that path to your Cursor **User Settings** (or `.cursor/settings.json`):

```json
{
  "agent_skills": [
    {
      "fullPath": "/your-project/.cursor/skills/ep-yf-report-skill/SKILL.md"
    }
  ]
}
```

---

## Use in Cursor chat

Once the skill is registered, ask the agent:

```
請使用 transcript_md_path=./input/ep645-260320.md，產出 Yahoo Finance charts 與 episode 走勢綜合報告。
```

The agent will:
1. Parse the transcript and extract stock tickers.
2. Download price data from Yahoo Finance.
3. Generate one `.svg` chart per ticker + a normalized comparison chart.
4. Write a complete markdown report to `output/`.

---

## Optional: Gemini LLM report

If you set `GEMINI_API_KEY` in a `.env` file, the agent can also generate an AI-written commentary section via `scripts/llm_gemini_report.py`.

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=models/gemini-2.0-flash
```

---

## License

MIT
