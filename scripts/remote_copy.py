import argparse
from pathlib import Path

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-file", required=True, type=Path)
    args = parser.parse_args()

    transport = paramiko.Transport((args.host, args.port))
    transport.connect(
        username=args.user,
        password=args.password_file.read_text(encoding="utf-8").strip(),
    )
    client = paramiko.SFTPClient.from_transport(transport)
    client.put(str(args.source), args.destination)
    client.close()
    transport.close()


if __name__ == "__main__":
    main()
