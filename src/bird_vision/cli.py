"""Command line interface for bird vision project."""

import typer
from pathlib import Path
from typing import Optional
import torch
import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from loguru import logger

from bird_vision.data.nabirds_dataset import NABirdsDataModule
from bird_vision.models.vision_model import VisionModel
from bird_vision.training.trainer import Trainer
from bird_vision.validation.model_validator import ModelValidator
from bird_vision.compression.model_compressor import ModelCompressor
from bird_vision.deployment.mobile_deployer import MobileDeployer
from bird_vision.deployment.raspberry_pi_deployer import RaspberryPiDeployer

app = typer.Typer(help="Bird Vision: Multi-modal bird identification system")
console = Console()


@app.command()
def train(
    config_path: str = typer.Option("configs/config.yaml", help="Path to config file"),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    resume_from: Optional[str] = typer.Option(None, help="Resume from checkpoint"),
) -> None:
    """Train a bird classification model."""
    console.print("[bold green]Starting training...[/bold green]")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    if experiment_name:
        cfg.experiment.name = experiment_name
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    
    # Setup data
    data_module = NABirdsDataModule(cfg.data)
    data_module.setup()
    
    # Setup model
    model = VisionModel(cfg.model)
    
    # Setup trainer
    trainer = Trainer(model, cfg, device)
    
    # Resume from checkpoint if specified
    if resume_from:
        checkpoint = trainer.checkpoint_manager.load_checkpoint(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        console.print(f"Resumed from checkpoint: {resume_from}")
    
    # Train
    trainer.fit(data_module.train_dataloader(), data_module.val_dataloader())
    
    console.print("[bold green]Training completed![/bold green]")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to model checkpoint"),
    config_path: str = typer.Option("configs/config.yaml", help="Path to config file"),
    save_results: bool = typer.Option(True, help="Save evaluation results"),
) -> None:
    """Evaluate a trained model."""
    console.print("[bold blue]Starting evaluation...[/bold blue]")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup data
    data_module = NABirdsDataModule(cfg.data)
    data_module.setup()
    
    # Load model
    model = VisionModel(cfg.model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluate
    validator = ModelValidator(cfg, device)
    results = validator.evaluate_model(
        model,
        data_module.test_dataloader(),
        model_name=Path(model_path).stem,
        save_results=save_results,
    )
    
    # Display results
    _display_evaluation_results(results)


@app.command()
def compress(
    model_path: str = typer.Argument(..., help="Path to model checkpoint"),
    config_path: str = typer.Option("configs/config.yaml", help="Path to config file"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
) -> None:
    """Compress a trained model for deployment."""
    console.print("[bold yellow]Starting model compression...[/bold yellow]")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    if output_dir:
        cfg.paths.models_dir = output_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VisionModel(cfg.model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Compress
    compressor = ModelCompressor(cfg)
    results = compressor.compress_model(model, sample_input, Path(model_path).stem)
    
    # Display results
    _display_compression_results(results)


@app.command()
def deploy(
    model_path: str = typer.Argument(..., help="Path to model checkpoint"),
    config_path: str = typer.Option("configs/config.yaml", help="Path to config file"),
    platform: str = typer.Option("mobile", help="Target platform (ios, android, mobile, esp32, raspberry_pi)"),
) -> None:
    """Deploy a model for mobile platforms."""
    console.print(f"[bold magenta]Preparing deployment for {platform}...[/bold magenta]")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    # Handle ESP32 platform
    if platform == "esp32":
        platform = "esp32_p4_eye"
        # Load ESP32-specific config
        cfg.deployment = hydra.compose(config_name="deployment/esp32")
    elif platform == "raspberry_pi":
        # Load Raspberry Pi-specific config
        cfg.deployment = hydra.compose(config_name="deployment/raspberry_pi")
    
    cfg.deployment.target_platform = platform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VisionModel(cfg.model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Deploy based on platform
    if platform == "esp32_p4_eye":
        from bird_vision.deployment.esp32_deployer import ESP32Deployer
        deployer = ESP32Deployer(cfg)
        results = deployer.prepare_for_esp32(model, sample_input, Path(model_path).stem)
        _display_esp32_deployment_results(results)
    elif platform == "raspberry_pi":
        deployer = RaspberryPiDeployer(cfg)
        output_dir = Path("deployments") / "raspberry_pi"
        output_dir.mkdir(parents=True, exist_ok=True)
        results = deployer.deploy_model(model, model_path, output_dir)
        _display_raspberry_pi_deployment_results(results)
    else:
        deployer = MobileDeployer(cfg)
        results = deployer.prepare_for_mobile(model, sample_input, Path(model_path).stem)
        _display_deployment_results(results, platform)


@app.command()
def pipeline(
    config_path: str = typer.Option("configs/config.yaml", help="Path to config file"),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    skip_training: bool = typer.Option(False, help="Skip training step"),
    model_path: Optional[str] = typer.Option(None, help="Use existing model (if skip_training)"),
    target_platform: str = typer.Option("mobile", help="Target deployment platform (mobile, esp32, raspberry_pi)"),
) -> None:
    """Run the complete pipeline: train -> validate -> compress -> deploy."""
    console.print("[bold cyan]Starting complete pipeline...[/bold cyan]")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    if experiment_name:
        cfg.experiment.name = experiment_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup data
    data_module = NABirdsDataModule(cfg.data)
    data_module.setup()
    
    if not skip_training:
        # 1. Train model
        console.print("\n[bold green]Step 1: Training model...[/bold green]")
        model = VisionModel(cfg.model)
        trainer = Trainer(model, cfg, device)
        trainer.fit(data_module.train_dataloader(), data_module.val_dataloader())
        
        # Get best checkpoint
        best_checkpoint_path = trainer.checkpoint_manager.get_best_checkpoint_path()
        model_path = str(best_checkpoint_path)
    else:
        if not model_path:
            console.print("[bold red]Error: model_path required when skipping training[/bold red]")
            return
        
        model = VisionModel(cfg.model)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # 2. Validate model
    console.print("\n[bold blue]Step 2: Validating model...[/bold blue]")
    validator = ModelValidator(cfg, device)
    validation_results = validator.evaluate_model(
        model, data_module.test_dataloader(), "pipeline_model"
    )
    _display_evaluation_results(validation_results)
    
    # 3. Compress model
    console.print("\n[bold yellow]Step 3: Compressing model...[/bold yellow]")
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    compressor = ModelCompressor(cfg)
    compression_results = compressor.compress_model(model, sample_input, "pipeline_model")
    _display_compression_results(compression_results)
    
    # 4. Deploy model
    console.print(f"\n[bold magenta]Step 4: Preparing {target_platform} deployment...[/bold magenta]")
    
    if target_platform == "esp32":
        # ESP32 deployment
        from bird_vision.deployment.esp32_deployer import ESP32Deployer
        esp32_cfg = cfg.copy()
        esp32_cfg.deployment = hydra.compose(config_name="deployment/esp32")
        deployer = ESP32Deployer(esp32_cfg)
        deployment_results = deployer.prepare_for_esp32(model, sample_input, "pipeline_model")
        _display_esp32_deployment_results(deployment_results)
    elif target_platform == "raspberry_pi":
        # Raspberry Pi deployment
        rpi_cfg = cfg.copy()
        rpi_cfg.deployment = hydra.compose(config_name="deployment/raspberry_pi")
        deployer = RaspberryPiDeployer(rpi_cfg)
        output_dir = Path("deployments") / "raspberry_pi"
        output_dir.mkdir(parents=True, exist_ok=True)
        deployment_results = deployer.deploy_model(model, model_path, output_dir)
        _display_raspberry_pi_deployment_results(deployment_results)
    else:
        # Mobile deployment
        deployer = MobileDeployer(cfg)
        deployment_results = deployer.prepare_for_mobile(model, sample_input, "pipeline_model")
        _display_deployment_results(deployment_results, target_platform)
    
    console.print("\n[bold cyan]Pipeline completed successfully![/bold cyan]")


def _display_evaluation_results(results: dict) -> None:
    """Display evaluation results in a table."""
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for metric, value in results["overall_metrics"].items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))
    
    console.print(table)


def _display_compression_results(results: dict) -> None:
    """Display compression results."""
    summary = results.get("compression_summary", {})
    
    table = Table(title="Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in summary.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)


def _display_deployment_results(results: dict, platform: str) -> None:
    """Display deployment results."""
    console.print(f"\n[bold green]Deployment packages created for {platform}:[/bold green]")
    
    packages = results.get("packages", {})
    for platform_name, package_path in packages.items():
        console.print(f"  • {platform_name}: {package_path}")


def _display_esp32_deployment_results(results: dict) -> None:
    """Display ESP32 deployment results."""
    console.print("\n[bold green]ESP32-P4-Eye deployment completed![/bold green]")
    
    if results.get("esp_dl_model", {}).get("success"):
        console.print("✅ ESP-DL model conversion successful")
    else:
        console.print("❌ ESP-DL model conversion failed")
    
    if results.get("firmware", {}).get("success"):
        console.print("✅ Firmware generation successful")
        firmware_dir = results["firmware"]["firmware_dir"]
        console.print(f"📁 Firmware location: {firmware_dir}")
        console.print("🔧 To build and flash:")
        console.print(f"   cd {firmware_dir}")
        console.print("   ./build.sh")
        console.print("   ./flash.sh")
    else:
        console.print("❌ Firmware generation failed")
    
    if results.get("package", {}).get("success"):
        package_dir = results["package"]["package_dir"]
        console.print(f"📦 Complete package: {package_dir}")
    
    # Display deployment info
    deployment_info = results.get("deployment_info", {})
    if deployment_info:
        model_stats = deployment_info.get("model_stats", {})
        constraints = deployment_info.get("constraints", {})
        
        table = Table(title="ESP32-P4 Deployment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Constraint", style="yellow")
        table.add_column("Status", style="green")
        
        # Model size
        size_mb = model_stats.get("size_mb", 0)
        max_size = constraints.get("max_size_mb", 8)
        size_status = "✅" if size_mb <= max_size else "⚠️"
        table.add_row("Model Size", f"{size_mb:.2f} MB", f"< {max_size} MB", size_status)
        
        # Inference time
        inference_ms = model_stats.get("avg_inference_time_ms", 0)
        max_inference = constraints.get("max_inference_ms", 200)
        inference_status = "✅" if inference_ms <= max_inference else "⚠️"
        table.add_row("Inference Time", f"{inference_ms:.1f} ms", f"< {max_inference} ms", inference_status)
        
        # Parameters
        params = model_stats.get("parameters", 0)
        table.add_row("Parameters", f"{params:,}", "-", "ℹ️")
        
        console.print(table)


def _display_raspberry_pi_deployment_results(results: dict) -> None:
    """Display Raspberry Pi deployment results."""
    console.print(f"\n[bold green]Raspberry Pi Deployment {'Success' if results.get('success') else 'Failed'}![/bold green]")
    
    if not results.get("success"):
        console.print("[bold red]Deployment failed with errors:[/bold red]")
        for error in results.get("errors", []):
            console.print(f"  - {error}")
        return
    
    # Deployment summary
    table = Table(title="Raspberry Pi Deployment Summary")
    table.add_column("Aspect", style="cyan")
    table.add_column("Details", style="magenta")
    
    table.add_row("Target Device", results.get("target_device", "rpi4"))
    table.add_row("Platform", results.get("platform", "raspberry_pi"))
    
    # Performance metrics
    performance = results.get("performance", {})
    if performance:
        size_mb = performance.get("model_size_mb", "N/A")
        if isinstance(size_mb, (int, float)):
            table.add_row("Model Size", f"{size_mb:.1f} MB")
        
        inference_time = performance.get("estimated_inference_time_ms", "N/A")
        if isinstance(inference_time, (int, float)):
            table.add_row("Est. Inference Time", f"{inference_time:.1f} ms")
        
        size_target = performance.get("meets_size_target", False)
        time_target = performance.get("meets_time_target", False)
        table.add_row("Meets Size Target", "✅ Yes" if size_target else "⚠️ No")
        table.add_row("Meets Time Target", "✅ Yes" if time_target else "⚠️ No")
    
    console.print(table)
    
    # Artifacts generated
    artifacts = results.get("artifacts", {})
    if artifacts:
        artifacts_table = Table(title="Generated Artifacts")
        artifacts_table.add_column("Artifact", style="cyan")
        artifacts_table.add_column("Location", style="yellow")
        
        for artifact_name, artifact_path in artifacts.items():
            display_name = artifact_name.replace("_", " ").title()
            artifacts_table.add_row(display_name, str(artifact_path))
        
        console.print(artifacts_table)
    
    # Installation instructions
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("1. Transfer the deployment package to your Raspberry Pi:")
    console.print("   [dim]scp -r deployments/raspberry_pi/raspberry_pi_package pi@your-pi-ip:~/[/dim]")
    console.print("2. On your Raspberry Pi, run the installation:")
    console.print("   [dim]cd ~/raspberry_pi_package && chmod +x install.sh && ./install.sh[/dim]")
    console.print("3. Start the service:")
    console.print("   [dim]sudo systemctl start bird-vision[/dim]")
    console.print("4. Check logs:")
    console.print("   [dim]sudo journalctl -u bird-vision -f[/dim]")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()