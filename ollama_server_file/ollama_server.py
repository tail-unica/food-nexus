from typing import Optional
import time
import httpx
import subprocess
import socket
import os

class OllamaServer:
    def __init__(self, base_port: int = 11434):
        self.base_port = base_port
        self.current_port = None
        self.process: Optional[subprocess.Popen] = None
        self.host = None

    def _find_available_port(self, start_port: int = 11434) -> Optional[int]:
        for port in range(start_port, start_port + 10000):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    continue
        return None

    def _is_server_responsive(self, host: str, timeout: int = 2) -> bool:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{host}/api/version")
                return response.status_code == 200
        except Exception:
            return False

    def start(self) -> str:
        if self.current_port:
            self._cleanup()

        port = self._find_available_port(self.base_port)
        if not port:
            raise RuntimeError("No available ports found")
        
        self.current_port = port
        self.host = f"http://127.0.0.1:{port}"
        
        self.process = subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ.copy(), "OLLAMA_HOST": f"127.0.0.1:{port}"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        start_time = time.time()
        while time.time() - start_time < 5:
            if self._is_server_responsive(self.host):
                return self.host
            time.sleep(0.5)

        raise TimeoutError("Server failed to start")

    def _cleanup(self) -> None:
        if self.process:
            self.process.terminate()
            self.process = None

        if self.current_port:
            self.current_port = None


class OllamaModel:
    def __init__(self, modelfile: str, model_name: Optional[str] = None) -> None:
        self.modelfile = modelfile
        self.model_name = model_name or self._extract_model_name() + "_custom"
        self.server = OllamaServer()
        self.host = self.server.start()
        self.client = httpx.Client(base_url=self.host)

        self._create_model()

    def _extract_model_name(self) -> str:
        for line in self.modelfile.splitlines():
            if line.startswith("FROM"):
                return line.split()[1]
        raise ValueError("Model name not found in modelfile.")

    def _create_model(self) -> None:
        response = self.client.post(
            "/api/models", json={"name": self.model_name, "modelfile": self.modelfile}
        )
        if response.status_code != 200:
            raise RuntimeError("Failed to create model")

    def generate(self, prompt: str, keep_alive: int = -1) -> str:
        response = self.client.post(
            f"/api/models/{self.model_name}/generate",
            json={"prompt": prompt, "keep_alive": keep_alive},
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise RuntimeError(f"Failed to generate response: {response.text}")
