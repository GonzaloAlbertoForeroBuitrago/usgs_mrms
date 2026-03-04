from __future__ import annotations

from pathlib import Path
import typer

from .config import PipelineConfig
from .logger import setup_logging, get_logger
from .pipeline import run_site

app = typer.Typer(add_completion=False, help="USGS → events → MRMS RadarOnly Zarr pipeline (resume-safe).")
log = get_logger("usgs_mrms_events.cli")

@app.command("run-site")
def run_site_cmd(
    site_id: str = typer.Argument(..., help="USGS site id (digits or 'USGS-XXXXXXXX')."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("data"), "--base-dir", help="Base output folder."),
    log_dir: Path | None = typer.Option(None, "--log-dir", help="Log folder (default: <base_dir>/logs)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs for this site."),
    upload: bool = typer.Option(False, "--upload", help="Upload results to S3 after completion."),
) -> None:
    """
    Run the full pipeline for a single site.
    """
    cfg = PipelineConfig(base_dir=base_dir.resolve(), log_dir=log_dir)
    paths = setup_logging(log_dir=cfg.log_dir)
    log.info(f"run-site | site_id={site_id} base_dir={cfg.base_dir} log={paths.run_log}")


    result = run_site(
        site_id=site_id,
        start_date=start,
        end_date=end,
        base_dir=base_dir,
        overwrite=overwrite,
        config=cfg,
        upload=upload
    )
    typer.echo(f"[{result['site_id']}] completed. Rain Zarr: {result['paths']['rain_zarr']}")


@app.command("run-many")
def run_many_cmd(
    sites_file: Path = typer.Argument(..., help="Text file with one site_id per line."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("data"), "--base-dir", help="Base output folder."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs per site."),
    upload: bool = typer.Option(False, "--upload", help="Upload results to S3 after completion."),
) -> None:
    """
    Run the pipeline for many sites from a file (one site per line).
    """
    from .pipeline import run_many

    site_ids = [ln.strip() for ln in sites_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = run_many(site_ids, start_date=start, end_date=end, base_dir=base_dir, overwrite=overwrite, upload=upload)
    typer.echo(out)

if __name__ == "__main__":
    app()