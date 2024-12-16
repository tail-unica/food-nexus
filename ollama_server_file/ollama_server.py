import atexit
import os
import subprocess
import ollama
import socket

ollama_servers = {}

#possibile start port da provare: 49152 con 65535-49152 max attempts
def find_available_port(start_port=11434, max_attempts=15000):
    for i in range(10):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found in the range.")

def start_ollama_server(host):
    try:
        process = subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ.copy(), "OLLAMA_HOST": host},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        ollama_servers[host] = process
    except Exception as e:
        print(f"Failed to start Ollama on port {host}: {e}")

def stop_ollama_server(host):
    if host in ollama_servers:
        process = ollama_servers[host]
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        del ollama_servers[host]
        print(f"Stopped Ollama server on port {host}.")

class OllamaModel:
    def __init__(self, model_name, modelfile, host=None):
        self.modelfile = modelfile
        self.model_name = model_name
        self.host = host or "127.0.0.1:11434"

        if "localhost" in self.host or "127.0.0.1" in self.host:
            self._initialize_server()

        self.client = ollama.Client(self.host)
        if self.model_name is None:
            self.model_name = self._extract_model_name() + "_custom"

        try:
            self.client.create(self.model_name, modelfile=self.modelfile)
        except Exception as e:
            print(f"Error creating model: {e}")
            self._handle_creation_error()

    def _extract_model_name(self):
        for line in self.modelfile.splitlines():
            if line.startswith("FROM"):
                return line.split()[1]
        raise ValueError("Model name not found in modelfile.")

    def _handle_creation_error(self):
        current_port = int(self.host.split(":")[-1])
        new_port = find_available_port(start_port=current_port + 1, max_attempts=10)
        new_host = f"127.0.0.1:{new_port}"
        print(f"Restarting server on {new_host}")
        stop_ollama_server(self.host)
        start_ollama_server(new_host)
        self.host = new_host
        self.client = ollama.Client(new_host)
        self.client.create(self.model_name, modelfile=self.modelfile)

    def _initialize_server(self):
        start_ollama_server(self.host)
        atexit.register(stop_ollama_server, self.host)

    def generate(self, prompt):
        try:
            return self.client.generate(self.model_name, prompt).response
        except Exception as e:
            print(f"Error generating response: {e}")
            self._handle_creation_error()
            return self.generate(prompt)
