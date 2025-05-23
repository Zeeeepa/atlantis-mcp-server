import atlantis


async def rls():
    """
    List remotes
    """

    await atlantis.client_command("\\remote refresh_all")
    await atlantis.client_command("\\remote list")

