import argparse
from pathlib import Path

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-file", required=True, type=Path)
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password_file.read_text(encoding="utf-8").strip(),
        timeout=15,
    )
    _, stdout, stderr = client.exec_command(args.command)
    for line in iter(stdout.readline, ""):
        print(line.rstrip("\r\n"))
    exit_code = stdout.channel.recv_exit_status()
    error = stderr.read().decode()
    client.close()
    if error:
        print(error, end="")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
