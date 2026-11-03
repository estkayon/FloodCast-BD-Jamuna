# tests/test_docker.py

"""Docker configuration tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestDockerfiles:
    """Verify Docker files exist and have required content."""

    def test_api_dockerfile_exists(self) -> None:
        path = ROOT / "docker" / "Dockerfile.api"
        assert path.exists(), "docker/Dockerfile.api missing"

    def test_dashboard_dockerfile_exists(self) -> None:
        path = ROOT / "docker" / "Dockerfile.dashboard"
        assert path.exists(), "docker/Dockerfile.dashboard missing"

    def test_api_dockerfile_has_healthcheck(self) -> None:
        path = ROOT / "docker" / "Dockerfile.api"
        content = path.read_text()
        assert "HEALTHCHECK" in content

    def test_api_dockerfile_exposes_port(self) -> None:
        path = ROOT / "docker" / "Dockerfile.api"
        content = path.read_text()
        assert "EXPOSE 8000" in content

    def test_dashboard_dockerfile_exposes_port(self) -> None:
        path = ROOT / "docker" / "Dockerfile.dashboard"
        content = path.read_text()
        assert "EXPOSE 8501" in content

    def test_api_dockerfile_sets_pythonpath(self) -> None:
        path = ROOT / "docker" / "Dockerfile.api"
        content = path.read_text()
        assert "PYTHONPATH" in content

    def test_dashboard_dockerfile_sets_api_url(self) -> None:
        path = ROOT / "docker" / "Dockerfile.dashboard"
        content = path.read_text()
        assert "FLOODCAST_API_URL" in content


class TestDockerCompose:
    """Verify docker-compose.yml structure."""

    def test_compose_file_exists(self) -> None:
        path = ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml missing"

    def test_compose_has_api_service(self) -> None:
        path = ROOT / "docker-compose.yml"
        content = path.read_text()
        assert "api:" in content

    def test_compose_has_dashboard_service(self) -> None:
        path = ROOT / "docker-compose.yml"
        content = path.read_text()
        assert "dashboard:" in content

    def test_compose_has_healthcheck(self) -> None:
        path = ROOT / "docker-compose.yml"
        content = path.read_text()
        assert "healthcheck:" in content


class TestCDWorkflow:
    """Verify CD workflow exists."""

    def test_cd_workflow_exists(self) -> None:
        path = ROOT / ".github" / "workflows" / "cd.yml"
        assert path.exists(), ".github/workflows/cd.yml missing"

    def test_cd_workflow_has_ghcr(self) -> None:
        path = ROOT / ".github" / "workflows" / "cd.yml"
        content = path.read_text()
        assert "ghcr.io" in content
