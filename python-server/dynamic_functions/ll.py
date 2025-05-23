import atlantis


async def ll():
    """
    List tools by last modified date
    """

    await atlantis.client_command("\\remote refresh_all")
    await atlantis.client_command("\\tool date")
