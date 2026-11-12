# Constrained SSH relay preflight

No relay, temporary agent, private-key read, credential forwarding or bulk copy was performed. The existing client on host 31112 is OpenSSH 8.8p1, which predates the required destination-constraint support, and the local known_hosts file has no saved key for the destination gateway port 31112. The check stopped before authentication as required. Existing transfers and remote workloads remain untouched.

The preceding private TCP probe failed on the exact tested route and port; see ../host_transfer_31112_20261112/readiness.json. This does not establish that every possible private route is unavailable.

No credentials, tokens or private keys are stored in this directory.
