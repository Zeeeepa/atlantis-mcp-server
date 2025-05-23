import atlantis


async def mls():
    """
    List MCP servers
    """

    await atlantis.client_command("\\remote refresh_all")
    await atlantis.client_command("\\mcp list")

