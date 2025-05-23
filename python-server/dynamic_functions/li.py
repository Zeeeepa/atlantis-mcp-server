import atlantis


async def li():
    """
    List tool details
    """

    await atlantis.client_command("\\remote refresh_all")
    await atlantis.client_command("\\tool info")
