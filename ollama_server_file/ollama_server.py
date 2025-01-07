"""
File containing script for create a ollama server and manage the reconnection 
to the server in a new port in case of error
"""


import atexit
import os
import subprocess
import ollama
import socket


ollama_servers = {}


#possible start port to try: 49152 with 65535-49152 max attempts#######
#idk well the client server teory
def find_available_port(start_port=11434, max_attempts=10000) -> int | None:
    """
    Function for searching an available port for the ollama server

    :param start_port: The port from which you start checking the first available one
    :param max_attempts: the number of port checked
    :return: the port available | None
    """
    for i in range(3):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found in the range.")


def start_ollama_server(host) -> None:
    """
    Function to start the ollama server

    :param host: the adress where to start the service
    :return: None
    """
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
    """
    Class for manage the ollama Model on a server
    """
    def __init__(self, model_name, modelfile, host=None) -> None:
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
        new_port = find_available_port(start_port=current_port + 1, max_attempts=10000)
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


    def generate(self, prompt) -> str:
        """
        Function to generate the model response

        :param prompt: the prompt to give to the model
        :return: the model rensponse
        """
        try:
            return self.client.generate(self.model_name, prompt)["response"]
        except Exception as e:
            print(f"Error generating response: {e}")
            self._handle_creation_error()
            return self.generate(prompt)
