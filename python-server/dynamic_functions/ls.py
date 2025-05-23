import atlantis


async def ls():
    """
    List tools and parameters
    """

    await atlantis.client_command("\\remote refresh_all")
    await atlantis.client_command("\\tool list")
