from __future__ import annotations

from pathlib import Path
import typer

from .pipeline import run_site

app = typer.Typer(add_completion=False, help="USGS → events → MRMS RadarOnly Zarr pipeline (resume-safe).")


@app.command("run-site")
def run_site_cmd(
    site_id: str = typer.Argument(..., help="USGS site id (digits or 'USGS-XXXXXXXX')."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("data"), "--base-dir", help="Base output folder."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs for this site."),
) -> None:
    """
    Run the full pipeline for a single site.
    """
    result = run_site(
        site_id=site_id,
        start_date=start,
        end_date=end,
        base_dir=base_dir,
        overwrite=overwrite,
    )
    typer.echo(f"[{result['site_id']}] completed. Rain Zarr: {result['paths']['rain_zarr']}")


@app.command("run-many")
def run_many_cmd(
    sites_file: Path = typer.Argument(..., help="Text file with one site_id per line."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("data"), "--base-dir", help="Base output folder."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs per site."),
) -> None:
    """
    Run the pipeline for many sites from a file (one site per line).
    """
    from .pipeline import run_many

    site_ids = [ln.strip() for ln in sites_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = run_many(site_ids, start_date=start, end_date=end, base_dir=base_dir, overwrite=overwrite)
    typer.echo(out)