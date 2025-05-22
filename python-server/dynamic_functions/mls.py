import atlantis


def mls():
    """
    List MCP servers
    """

    atlantis.client_command("\\remote refresh_all")
    atlantis.client_command("\\mcp list")

