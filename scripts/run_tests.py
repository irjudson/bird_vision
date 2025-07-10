#!/usr/bin/env python3
"""Comprehensive test runner for Bird Vision project."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json

class TestRunner:
    """Comprehensive test runner with different test suites."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
    
    def run_unit_tests(self, verbose: bool = False) -> bool:
        """Run unit tests."""
        print("🧪 Running unit tests...")
        
        cmd = [
            "pytest", 
            "tests/unit/", 
            "--cov=bird_vision",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        success = result.returncode == 0
        self.results["unit_tests"] = {
            "success": success,
            "duration": duration,
            "command": " ".join(cmd)
        }
        
        if success:
            print(f"✅ Unit tests passed in {duration:.2f}s")
        else:
            print(f"❌ Unit tests failed in {duration:.2f}s")
        
        return success
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = ["pytest", "tests/integration/", "--timeout=300"]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        success = result.returncode == 0
        self.results["integration_tests"] = {
            "success": success,
            "duration": duration,
            "command": " ".join(cmd)
        }
        
        if success:
            print(f"✅ Integration tests passed in {duration:.2f}s")
        else:
            print(f"❌ Integration tests failed in {duration:.2f}s")
        
        return success
    
    def run_e2e_tests(self, verbose: bool = False) -> bool:
        """Run end-to-end tests."""
        print("🌐 Running end-to-end tests...")
        
        cmd = ["pytest", "tests/e2e/", "--timeout=600"]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        success = result.returncode == 0
        self.results["e2e_tests"] = {
            "success": success,
            "duration": duration,
            "command": " ".join(cmd)
        }
        
        if success:
            print(f"✅ End-to-end tests passed in {duration:.2f}s")
        else:
            print(f"❌ End-to-end tests failed in {duration:.2f}s")
        
        return success
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")
        
        cmd = [
            "pytest", 
            "tests/benchmarks/", 
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-json=benchmark_results.json"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        success = result.returncode == 0
        self.results["performance_tests"] = {
            "success": success,
            "duration": duration,
            "command": " ".join(cmd)
        }
        
        if success:
            print(f"✅ Performance benchmarks completed in {duration:.2f}s")
            
            # Load and display benchmark results
            benchmark_file = self.project_root / "benchmark_results.json"
            if benchmark_file.exists():
                self._display_benchmark_results(benchmark_file)
        else:
            print(f"❌ Performance benchmarks failed in {duration:.2f}s")
        
        return success
    
    def run_code_quality_checks(self, verbose: bool = False) -> bool:
        """Run code quality checks (linting, formatting, type checking)."""
        print("🔍 Running code quality checks...")
        
        checks = [
            ("Black (formatting)", ["black", "--check", "src/", "tests/"]),
            ("isort (imports)", ["isort", "--check-only", "src/", "tests/"]),
            ("flake8 (linting)", ["flake8", "src/", "tests/", "--max-line-length=88", "--extend-ignore=E203,W503"]),
            ("mypy (type checking)", ["mypy", "src/bird_vision", "--ignore-missing-imports"]),
        ]
        
        all_passed = True
        check_results = {}
        
        for check_name, cmd in checks:
            print(f"  Running {check_name}...")
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=not verbose)
            duration = time.time() - start_time
            
            success = result.returncode == 0
            check_results[check_name] = {
                "success": success,
                "duration": duration,
                "command": " ".join(cmd)
            }
            
            if success:
                print(f"    ✅ {check_name} passed")
            else:
                print(f"    ❌ {check_name} failed")
                if verbose and result.stdout:
                    print(f"    Output: {result.stdout.decode()}")
                if verbose and result.stderr:
                    print(f"    Error: {result.stderr.decode()}")
                all_passed = False
        
        self.results["code_quality"] = {
            "success": all_passed,
            "checks": check_results
        }
        
        return all_passed
    
    def run_cli_tests(self, verbose: bool = False) -> bool:
        """Test CLI functionality."""
        print("💻 Testing CLI functionality...")
        
        cli_tests = [
            ("CLI help", ["bird-vision", "--help"]),
            ("CLI import", ["python", "-c", "from bird_vision.cli import main; print('CLI import successful')"]),
        ]
        
        all_passed = True
        cli_results = {}
        
        for test_name, cmd in cli_tests:
            print(f"  Running {test_name}...")
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=not verbose)
            duration = time.time() - start_time
            
            success = result.returncode == 0
            cli_results[test_name] = {
                "success": success,
                "duration": duration,
                "command": " ".join(cmd)
            }
            
            if success:
                print(f"    ✅ {test_name} passed")
            else:
                print(f"    ❌ {test_name} failed")
                all_passed = False
        
        self.results["cli_tests"] = {
            "success": all_passed,
            "tests": cli_results
        }
        
        return all_passed
    
    def run_all_tests(self, verbose: bool = False, skip_slow: bool = False) -> bool:
        """Run all test suites."""
        print("🚀 Running complete test suite...")
        print("=" * 60)
        
        all_passed = True
        
        # Code quality first (fast feedback)
        if not self.run_code_quality_checks(verbose):
            all_passed = False
        
        print()
        
        # CLI tests
        if not self.run_cli_tests(verbose):
            all_passed = False
        
        print()
        
        # Unit tests
        if not self.run_unit_tests(verbose):
            all_passed = False
        
        print()
        
        # Integration tests
        if not skip_slow and not self.run_integration_tests(verbose):
            all_passed = False
        
        print()
        
        # E2E tests
        if not skip_slow and not self.run_e2e_tests(verbose):
            all_passed = False
        
        print()
        
        # Performance tests (optional)
        if not skip_slow:
            self.run_performance_tests(verbose)  # Don't fail on performance tests
        
        return all_passed
    
    def _display_benchmark_results(self, benchmark_file: Path):
        """Display benchmark results summary."""
        try:
            with open(benchmark_file) as f:
                data = json.load(f)
            
            print("\n📊 Performance Benchmark Summary:")
            print("-" * 40)
            
            if "benchmarks" in data:
                for benchmark in data["benchmarks"][:5]:  # Show top 5
                    name = benchmark.get("name", "Unknown")
                    mean_time = benchmark.get("stats", {}).get("mean", 0)
                    print(f"  {name}: {mean_time:.4f}s")
        
        except Exception as e:
            print(f"Could not parse benchmark results: {e}")
    
    def generate_report(self, output_file: str = "test_report.json"):
        """Generate test report."""
        total_duration = sum(
            result.get("duration", 0) for result in self.results.values() 
            if isinstance(result, dict) and "duration" in result
        )
        
        report = {
            "timestamp": time.time(),
            "total_duration": total_duration,
            "summary": {
                "total_suites": len(self.results),
                "passed_suites": sum(1 for r in self.results.values() if r.get("success", False)),
                "failed_suites": sum(1 for r in self.results.values() if not r.get("success", True)),
            },
            "results": self.results
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Test report saved to {output_file}")
        return report
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📋 Test Summary")
        print("=" * 60)
        
        for suite_name, result in self.results.items():
            if isinstance(result, dict) and "success" in result:
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                duration = result.get("duration", 0)
                print(f"{suite_name:20} {status:10} ({duration:.2f}s)")
        
        total_suites = len([r for r in self.results.values() if isinstance(r, dict) and "success" in r])
        passed_suites = sum(1 for r in self.results.values() if isinstance(r, dict) and r.get("success", False))
        
        print("-" * 60)
        print(f"Total: {passed_suites}/{total_suites} test suites passed")
        
        if passed_suites == total_suites:
            print("🎉 All tests passed!")
        else:
            print("💥 Some tests failed!")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Run Bird Vision test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests (integration, e2e)")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e-only", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--quality-only", action="store_true", help="Run only code quality checks")
    parser.add_argument("--cli-only", action="store_true", help="Run only CLI tests")
    parser.add_argument("--report", default="test_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    runner = TestRunner(project_root)
    
    success = True
    
    try:
        if args.unit_only:
            success = runner.run_unit_tests(args.verbose)
        elif args.integration_only:
            success = runner.run_integration_tests(args.verbose)
        elif args.e2e_only:
            success = runner.run_e2e_tests(args.verbose)
        elif args.performance_only:
            success = runner.run_performance_tests(args.verbose)
        elif args.quality_only:
            success = runner.run_code_quality_checks(args.verbose)
        elif args.cli_only:
            success = runner.run_cli_tests(args.verbose)
        else:
            success = runner.run_all_tests(args.verbose, args.skip_slow)
        
        runner.print_summary()
        runner.generate_report(args.report)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test run interrupted by user")
        success = False
    except Exception as e:
        print(f"\n💥 Test run failed with error: {e}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()