
from __future__ import annotations
import json, os, sys
from pathlib import Path
import click
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
load_dotenv()
console = Console()
RISK_COLOUR = {"High": "red", "Medium": "yellow", "Low": "green"}
RISK_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
CLASS_COLOUR = {"Glioma": "bold red", "Meningioma": "bold yellow", "No Tumor": "bold green", "Pituitary": "bold cyan"}

def _conf_bar(pct, width=28):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)

def _print_header():
    console.print(Panel.fit("[bold cyan]🧠 NeuroScan Agent[/bold cyan]\n[dim]IEEE SCEECS 2026 | Brain Tumor Classification | Powered by Gemini (FREE)[/dim]", border_style="cyan"))

def _print_result(result):
    r = result.get("result", {})
    rv = result.get("review", {})
    inp = result.get("input", {})
    trace = result.get("agent_trace", {})
    classification = r.get("classification", "Unknown")
    risk = r.get("risk", "Unknown")
    conf = r.get("confidence", {})
    top_conf = r.get("top_confidence") or 0
    flagged = rv.get("flagged", False)
    flag_severity = rv.get("flag_severity", "")
    flag_reason = rv.get("flag_reason", "")
    risk_colour = RISK_COLOUR.get(risk, "white")
    risk_icon = RISK_ICON.get(risk, "⚪")
    cls_colour = CLASS_COLOUR.get(classification, "bold white")
    summary_lines = [f"[{cls_colour}]  {classification}[/{cls_colour}]", f"  Risk: [{risk_colour}]{risk_icon} {risk}[/{risk_colour}]", f"  Top confidence: [bold]{top_conf:.1f}%[/bold]", f"  Image: [dim]{inp.get('image_filename', '')}[/dim]", f"  Session: [dim]{result.get('session_id', '')}[/dim]"]
    if flagged:
        summary_lines.append(f"\n  [bold yellow]⚠ FLAGGED FOR REVIEW[/bold yellow] [dim]({flag_severity} priority)[/dim]")
    console.print(Panel("\n".join(summary_lines), title="[bold]RESULT[/bold]", border_style=risk_colour, padding=(1, 2)))
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    tbl.add_column("Class", min_width=12)
    tbl.add_column("Confidence", min_width=32)
    tbl.add_column("Score", justify="right")
    for cls in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
        pct = conf.get(cls, 0.0)
        colour = CLASS_COLOUR.get(cls, "white") if cls == classification else "white"
        marker = " ◄" if cls == classification else ""
        tbl.add_row(f"[{colour}]{cls}[/{colour}]", f"[{colour}]{_conf_bar(pct)}[/{colour}]", f"[{colour}]{pct:.1f}%{marker}[/{colour}]")
    console.print(tbl)
    notes = r.get("clinical_notes")
    if notes:
        console.print(Panel(notes, title="[bold]Clinical Reasoning[/bold]", border_style="dim", padding=(0, 2)))
    if flagged and flag_reason:
        console.print(Panel(f"[yellow]{flag_reason}[/yellow]", title=f"[bold yellow]⚠ Review Flag — {flag_severity.upper()} Priority[/bold yellow]", border_style="yellow", padding=(0, 2)))
    rec = r.get("recommendation")
    if rec:
        console.print(Panel(rec, title="[bold]Recommendation[/bold]", border_style="dim", padding=(0, 2)))
    summary = trace.get("final_summary")
    if summary:
        console.print(Panel(f"[dim]{summary}[/dim]", title="[bold]Agent Summary[/bold]", border_style="dim", padding=(0, 2)))
    reports = result.get("reports", {})
    if reports.get("json"):
        console.print(f"\n  [dim]📄 JSON:     {reports['json']}[/dim]")
    if reports.get("markdown"):
        console.print(f"  [dim]📝 Markdown: {reports['markdown']}[/dim]")
    console.print("\n[dim italic]⚠ Research tool only. Not for clinical use. Consult a licensed radiologist.[/dim italic]\n")

@click.group()
def cli():
    """NeuroScan Agent — Brain Tumor MRI Analysis"""
    pass

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, file_okay=True))
@click.option("--context", "-c", default="", help="Patient context")
@click.option("--session-id", "-s", default="", help="Override session ID")
@click.option("--mode", "-m", type=click.Choice(["api", "local"]), default=None)
@click.option("--json-only", is_flag=True, help="Print raw JSON")
def analyze(image_path, context, session_id, mode, json_only):
    """Analyse a brain MRI image.\n\n\b\nExamples:\n    python cli.py analyze brain.jpg\n    python cli.py analyze brain.jpg --context "45F, headaches"\n    python cli.py analyze brain.jpg --json-only"""
    if not json_only:
        _print_header()
        console.print(f"  [dim]Image: {Path(image_path).resolve()}[/dim]")
        if context:
            console.print(f"  [dim]Context: {context}[/dim]")
        console.print()
    if mode:
        os.environ["ANALYSIS_MODE"] = mode
    try:
        from agent.planner import NeuroScanPlanner
        planner = NeuroScanPlanner()
    except ValueError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)
    result = None
    error = None
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task("Running agent analysis...", total=None)
        try:
            result = planner.run(image_path=image_path, patient_context=context, session_id=session_id or None)
        except Exception as exc:
            error = exc
    if error:
        console.print(f"\n[bold red]Error:[/bold red] {error}")
        if "GEMINI_API_KEY" in str(error):
            console.print("[dim]Tip: Add GEMINI_API_KEY to .env — free at https://aistudio.google.com/apikey[/dim]")
        elif "OPENROUTER_API_KEY" in str(error):
            console.print("[dim]Tip: Add OPENROUTER_API_KEY to .env — free at https://openrouter.ai/keys[/dim]")
        sys.exit(1)
    if json_only:
        click.echo(json.dumps(result, indent=2))
    else:
        _print_result(result)

@cli.command()
@click.argument("session_id")
@click.option("--md", is_flag=True, help="Show Markdown report")
def report(session_id, md):
    """View a saved diagnostic report by session ID."""
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    path = output_dir / session_id / ("report.md" if md else "report.json")
    if not path.exists():
        console.print(f"[red]No report found for session {session_id}[/red]")
        sys.exit(1)
    if md:
        console.print(path.read_text(encoding="utf-8"))
    else:
        _print_header()
        _print_result(json.loads(path.read_text(encoding="utf-8")))

@cli.command()
def queue():
    """List scans flagged for radiologist review."""
    qf = Path(os.getenv("OUTPUT_DIR", "outputs")) / "review_queue.jsonl"
    if not qf.exists():
        console.print("[green]✓ Review queue is empty.[/green]")
        return
    entries = []
    for line in qf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        console.print("[green]✓ Review queue is empty.[/green]")
        return
    _print_header()
    tbl = Table(title=f"Review Queue ({len(entries)} scans)", box=box.ROUNDED)
    tbl.add_column("Session ID", style="dim")
    tbl.add_column("Image")
    tbl.add_column("Classification")
    tbl.add_column("Confidence", justify="right")
    tbl.add_column("Severity")
    tbl.add_column("Reason")
    sev_colour = {"high": "red", "medium": "yellow", "low": "cyan"}
    for e in entries:
        sev = e.get("severity", "low")
        tbl.add_row(e.get("session_id", ""), e.get("image", ""), e.get("classification", ""), f"{e.get('top_confidence', 0):.1f}%", f"[{sev_colour.get(sev, 'white')}]{sev.upper()}[/{sev_colour.get(sev, 'white')}]", (e.get("reason", "")[:60] + "…") if len(e.get("reason", "")) > 60 else e.get("reason", ""))
    console.print(tbl)

@cli.command()
def sessions():
    """List all saved analysis sessions."""
    od = Path(os.getenv("OUTPUT_DIR", "outputs"))
    if not od.exists():
        console.print("[dim]No sessions found.[/dim]")
        return
    rows = []
    for sd in sorted(od.iterdir(), reverse=True):
        rp = sd / "report.json"
        if rp.exists():
            try:
                rows.append(json.loads(rp.read_text(encoding="utf-8")))
            except Exception:
                continue
    if not rows:
        console.print("[dim]No sessions found.[/dim]")
        return
    _print_header()
    tbl = Table(title=f"Sessions ({len(rows)})", box=box.ROUNDED)
    tbl.add_column("Session ID", style="dim")
    tbl.add_column("Date")
    tbl.add_column("Image")
    tbl.add_column("Classification")
    tbl.add_column("Risk")
    tbl.add_column("Flagged")
    for data in rows:
        risk = data.get("result", {}).get("risk", "")
        rc = RISK_COLOUR.get(risk, "white")
        flagged = data.get("review", {}).get("flagged", False)
        tbl.add_row(data.get("session_id", ""), (data.get("started_at", "") or "")[:19].replace("T", " "), data.get("input", {}).get("image_filename", ""), data.get("result", {}).get("classification", ""), f"[{rc}]{risk}[/{rc}]", "[yellow]⚠ Yes[/yellow]" if flagged else "[dim]No[/dim]")
    console.print(tbl)

if __name__ == "__main__":
    cli()
