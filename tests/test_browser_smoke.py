from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


class BrowserSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        health_url = f"http://127.0.0.1:{cls.port}/health"
        for _ in range(50):
            try:
                with urlopen(health_url, timeout=0.2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.1)

        cls.server.terminate()
        raise RuntimeError("Uvicorn did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=5)

    @unittest.skipUnless(CHROME.is_file(), "Chrome is required for browser smoke testing")
    def test_default_trajectory_is_rendered_by_javascript(self):
        with tempfile.TemporaryDirectory() as profile:
            result = subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--disable-gpu",
                    "--virtual-time-budget=3000",
                    f"--user-data-dir={profile}",
                    "--dump-dom",
                    f"http://127.0.0.1:{self.port}/",
                ],
                check=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )

        html = result.stdout
        self.assertRegex(html, r'id="position-value">-0\.502<')
        self.assertRegex(html, r'id="velocity-value">0\.939<')
        self.assertRegex(html, r'id="spring-path"[^>]*d="M[^\"]+"')
        self.assertRegex(html, r'id="trajectory-path"[^>]*d="M[^\"]+"')


if __name__ == "__main__":
    unittest.main()
